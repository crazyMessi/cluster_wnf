"""
本文件实现以下功能，每个子功能均用一个函数实现：
0. 计算指定分辨率的 winding number 场
1. 使用 scikit-image 的 SLIC 算法将 winding number 场划分为超体素(wn_grid)
2. 使用类似 compute_G_fast 的方法构造超体素网络
3. 使用谱聚类对超体素网络进行二分
请确保已安装所需依赖包:numpy, open3d, igl, scikit-image, scikit-learn, cc3d
"""

import numpy as np
import open3d as o3d
from igl.copyleft.cgal import fast_winding_number
from skimage.segmentation import slic
from sklearn.cluster import SpectralClustering
import networkx as nx
from cal_wnf import WNF
from tools import visualize_partition, plot_partition,normalize_points



def estimate_normals(pcd, neighbors=30):
    """
    使用 PCA 估计点云法向量，并保证法向量方向一致
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighbors))
    pcd.orient_normals_consistent_tangent_plane(10)
    return np.asarray(pcd.normals)

def create_uniform_grid(resolution):
    """
    创建均匀采样的 3D 网格采样点
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    z = np.linspace(0, 1, resolution)
    zz, yy, xx = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    grid_shape = (resolution, resolution, resolution)
    return points, grid_shape

'''
使用 scikit-image 的 SLIC 算法对 winding number 场进行分割
@param wn_field: 3D winding number 场（numpy 数组）
@param n_segments: 期望的超体素数量
@param compactness: SLIC 算法的紧凑性参数
@return segmented_grid: 分割后的标签网格，形状与 wn_field 相同
'''
def segment_winding_field(wn_field: np.ndarray, n_segments: int = 100, compactness: float = 0.1):
    """
    使用 SLIC 算法对 winding number 场进行分割
    参数:
      wn_field: 3D winding number 场（numpy 数组）
      n_segments: 期望的超体素数量
      compactness: SLIC 算法的紧凑性参数
    返回:
      segmented_grid: 分割后的标签网格，每个区域为一个超体素
    """
    segments = slic(wn_field, n_segments=n_segments, compactness=compactness, channel_axis=None, start_label=0)
    segments_flat = segments.flatten()
    wn_flat = wn_field.flatten()
    segments_sum = {}
    for i in range(segments_flat.max() + 1):
        mask = segments_flat == i
        wn_sum = np.sum(wn_flat[mask])
        segments_sum[i] = wn_sum
    # 根据 wn_sum 对 labels 进行排序
    segments_sum = sorted(segments_sum.items(), key=lambda x: x[1], reverse=True)
    new_segments = np.zeros_like(segments)
    for i, (label, _) in enumerate(segments_sum):
        new_segments[segments == label] = i
    new_segments = new_segments.reshape(wn_field.shape)
    return new_segments

# 2. 使用类似 compute_G_fast 的方法构造超体素网络
def compute_supervoxel_network(segmented_grid, wn_field):
    """
    构造超体素网络的邻接矩阵
    参数:
      segmented_grid: 分割后的超体素标签网格
      wn_field: 原始的 winding number 场（3D numpy 数组）
    返回:
      G: 超体素网络邻接矩阵，形状为 [K, K]，K 为超体素数量
    """
    K = len(np.unique(segmented_grid))
    G = np.zeros((K, K), dtype=np.float32)
    visited = np.zeros_like(G)
    boundary_mask = np.zeros_like(segmented_grid,dtype=np.int32)
    # 针对每个轴以及正负方向，统计不同超体素间的边权
    for axis in range(3):
        for shift in [1, -1]:
            shifted = np.roll(segmented_grid, shift, axis=axis)
            shifted_wn = np.roll(wn_field, shift, axis=axis)
            mask = (segmented_grid != shifted)
            # 边界处理，避免错误连接
            if shift == 1:
                if axis == 0:
                    mask[-1, :, :] = False
                elif axis == 1:
                    mask[:, -1, :] = False
                elif axis == 2:
                    mask[:, :, -1] = False
            else:
                if axis == 0:
                    mask[0, :, :] = False
                elif axis == 1:
                    mask[:, 0, :] = False
                elif axis == 2:
                    mask[:, :, 0] = False

            current_labels = segmented_grid[mask]
            neighbor_labels = shifted[mask]
            # 计算winding number的差值的绝对值
            wn_diff = np.abs(wn_field[mask] - shifted_wn[mask])
            visited[current_labels, neighbor_labels] = 1
            visited[neighbor_labels, current_labels] = 1
            G[current_labels.flatten(), neighbor_labels.flatten()] += wn_diff.flatten()
            G[neighbor_labels.flatten(), current_labels.flatten()] += wn_diff.flatten() # 保证对称性
            boundary_mask[mask] += 1
            
    G[visited==1] += 0.1
    nG = nx.from_numpy_array(G)
    components = list(nx.connected_components(nG))
    assert len(components) == 1, "G 的联通分量数量不为 1"
    return G

# 3. 使用谱聚类对超体素网络进行二分
def spectral_bisect_network(G):
    """
    利用谱聚类对超体素网络进行二分
    参数:
      G: 超体素网络的邻接矩阵（形状为 [K, K]）
    返回:
      clusters: 每个超体素的二分标签（0 或 1），长度为 K
    """
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=0)    
    sp_G = G.copy()
    # 非零部分取倒数
    sp_G[G!=0] = 1/G[G!=0]
    clusters = spectral.fit_predict(sp_G)
    return clusters

def field_to_clusters(field_data: np.ndarray, n_segments: int = 100, compactness: float = 0.1) -> np.ndarray:
    """
    将3D场数据转换为二分聚类结果的端到端API
    
    Args:
        field_data: (N, N, N) array - 输入的3D场数据
        n_segments: int - 超体素分割的数量
        compactness: float - SLIC算法的紧凑性参数
    
    Returns:
        (N, N, N) array - 二分聚类结果，每个体素的值为0或1
    """
    # 1. 超体素分割
    segmented_grid = segment_winding_field(field_data, n_segments, compactness)
    
    # 2. 构建超体素网络
    G = compute_supervoxel_network(segmented_grid, field_data)
    
    # 3. 谱聚类
    clusters = spectral_bisect_network(G)
    
    # 4. 将聚类结果映射回体素网格 to accelerate
    cluster_grid = np.zeros_like(field_data)
    for i in range(len(clusters)):
        cluster_grid[segmented_grid == i] = clusters[i]
    return cluster_grid

def field_to_clusters_with_vis(field_data: np.ndarray, n_segments: int = 100, 
                             compactness: float = 0.1, save_path: str = None) -> tuple:
    """
    带可视化的端到端API
    
    Args:
        field_data: (N, N, N) array - 输入的3D场数据
        n_segments: int - 超体素分割的数量
        compactness: float - SLIC算法的紧凑性参数
        save_path: str - 可视化结果保存路径，如果为None则不保存
    
    Returns:
        tuple:
            - (N, N, N) array - 二分聚类结果
            - (N, N, N) array - 超体素分割结果
            - (K, K) array - 超体素网络邻接矩阵
            - (K,) array - 聚类标签
    """
    # 1. 超体素分割
    print("Step 1: Supervoxel segmentation...")
    segmented_grid = segment_winding_field(field_data, n_segments, compactness)
    
    # 2. 构建超体素网络
    print("Step 2: Building supervoxel network...")
    G = compute_supervoxel_network(segmented_grid, field_data)
    
    # 3. 谱聚类
    print("Step 3: Spectral clustering...")
    clusters = spectral_bisect_network(G)
    
    # 4. 将聚类结果映射回体素网格
    print("Step 4: Mapping clusters back to grid...")
    cluster_grid = np.zeros_like(field_data)
    for i in range(len(clusters)):
        cluster_grid[segmented_grid == i] = clusters[i]
    
    # 5. 可视化（如果需要）
    if save_path is not None:
        print("Step 5: Saving visualizations...")
        # 保存原始场数据
        plot_partition(field_data, f"{save_path}_field.png")
        # 保存超体素分割结果
        plot_partition(segmented_grid, f"{save_path}_segments.png")
        # 保存聚类结果
        plot_partition(cluster_grid, f"{save_path}_clusters.png")
    
    return cluster_grid, segmented_grid, G, clusters

# 使用示例
if __name__ == "__main__":
    # 创建一个示例场数据
    N = 50
    test_field = np.random.rand(N, N, N)

    clusters, segments, G, labels = field_to_clusters_with_vis(
        test_field,
        n_segments=100,
        compactness=0.1,
        save_path="./output/test"
    )
    print("\nDetailed clustering complete:")
    print("- Cluster grid shape:", clusters.shape)
    print("- Number of supervoxels:", len(np.unique(segments)))
    print("- Network size:", G.shape)
    print("- Cluster distribution:", np.bincount(labels))


    
    
