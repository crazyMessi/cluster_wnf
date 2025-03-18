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
from skimage.segmentation import slic
from sklearn.cluster import SpectralClustering
import networkx as nx
from cal_wnf import WNF
from tools import visualize_partition, plot_partition,normalize_points
import tools
from scipy.spatial import cKDTree



def estimate_normals(pcd, neighbors=30):
    """
    使用 PCA 估计点云法向量，并保证法向量方向一致
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighbors))
    pcd.orient_normals_consistent_tangent_plane(10)
    return np.asarray(pcd.normals)


def my_dust(labels,threshold):
    '''
    对已经聚类好的grid进行dust操作
    '''    
    count = np.bincount(labels.flatten())
    for i in np.unique(labels):
        if count[i] < threshold:
            labels[labels==i] = 0
    return labels

def cc3d_segmentatino(wn_field,level_count=10,min_size=100):
    """
    使用 cc3d 对 winding number 场进行分割
    @param level_count: 分割的层数
    @param min_size: 最小连通域大小
    """
    import cc3d
    from scipy import ndimage
    connectivity = 26
    levelized_field = np.zeros_like(wn_field)
    # 
    
    min_val = np.min(wn_field)
    max_val = np.max(wn_field)
    for i in range(level_count):
        levelized_field[wn_field >= min_val + i * (max_val - min_val) / level_count] = i+1
    labels = cc3d.connected_components(levelized_field,connectivity=connectivity)
    tools.plot_partition(labels,save_path="./temp/cc3d_levelized.png")
    # 去除小于min_size的连通域
    # labels = cc3d.dust(labels,threshold=min_size,connectivity=connectivity)
    labels = my_dust(labels,min_size)
    tools.plot_partition(labels,save_path="./temp/cc3d_after_dust.png")
    # 使用opencv2膨胀, 去除孤立的连通域
    kernel = np.zeros((3,3,3))
    # 只有上下左右前后六个方向为1
    kernel[1,:,:] = 1
    kernel[:,1,:] = 1
    kernel[:,:,1] = 1
    kernel[0,:,:] = 1
    kernel[:,0,:] = 1
    kernel[:,:,0] = 1
    kernel[1,1,1] = 1
    while np.any(labels==0):
        labels = ndimage.grey_dilation(labels,footprint=kernel).astype(np.int32)
    return labels

def reallcate_labels_by_mean_val(labels,values):
    """
    根据 values 的均值重新分配 labels
    """
    l_sum = {}
    l_size = {}
    l_mean = {}
    for i in range(labels.max() + 1):
        tmask = labels == i
        l_sum[i] = np.sum(values[tmask])
        l_size[i] = np.sum(tmask)
        l_mean[i] = np.mean(values[tmask])
    # 根据 l_mean 对 labels 进行排序
    l_mean = sorted(l_mean.items(), key=lambda x: x[1], reverse=False)
    # 去除nan
    l_mean = [x for x in l_mean if not np.isnan(x[1])]
    new_labels = np.zeros_like(labels)
    for i, (label, _) in enumerate(l_mean):
        new_labels[labels == label] = i
    id_list = np.unique(new_labels)
    segment_mean = np.zeros(len(id_list))
    assert(len(id_list) == np.max(id_list)+1)
    for id in id_list:
        segment_mean[id] = np.mean(values[new_labels == id])
    sorted_segment_mean = np.sort(segment_mean)
    assert np.all(sorted_segment_mean == segment_mean), "segment_mean 排序后不一致"
    return new_labels,segment_mean

'''
使用 scikit-image 的 SLIC 算法对 winding number 场进行分割
@param wn_field: 3D winding number 场（numpy 数组）
@param n_segments: 期望的超体素数量
@param compactness: SLIC 算法的紧凑性参数
@return segmented_grid: 分割后的标签网格，形状与 wn_field 相同
'''
def segment_winding_field(wn_field: np.ndarray, n_segments: int = 100, compactness: float = 0.1, mask: np.ndarray = None):
    """
    使用 SLIC 算法对 winding number 场进行分割
    参数:
      wn_field: 3D winding number 场（numpy 数组）
      n_segments: 期望的超体素数量
      compactness: SLIC 算法的紧凑性参数
    返回:
      segmented_grid: 分割后的标签网格，每个区域为一个超体素
    """
    segments = slic(wn_field, n_segments=n_segments, compactness=compactness, channel_axis=None, start_label=0, mask=mask)
    # segments_sum = {}
    # segments_size = {}
    # segments_mean = {} 
    # assert segments_flat.min() == 0, "segments_flat 的最小值不为 0"
    # for i in range(segments_flat.max() + 1):
    #     tmask = segments_flat == i
    #     wn_i = wn_flat[tmask]
    #     segments_sum[i] = np.sum(wn_i)
    #     segments_size[i] = len(wn_i)
    #     segments_mean[i] = np.mean(wn_i)    
    # # 根据 wn_sum 对 labels 进行排序
    # segments_sum = sorted(segments_sum.items(), key=lambda x: x[1], reverse=True)
    # new_segments_flat = np.zeros(len(segments_flat))
    # for i, (label, _) in enumerate(segments_sum):
    #     new_segments_flat[segments_flat == label] = i
    # # new_segments = new_segments_flat.reshape(wn_field.shape)
    # new_segments = segments.copy()
    # new_segments[mask] = new_segments_flat
    # new_segments = reallcate_labels_by_mean_val(segments,wn_field)
    return segments


# class CubuField:
#     def __init__(self,shape,bbox,points):
#         self.shape = shape
#         self.bbox = bbox
#         self.points = points

    
#     def get_point_counts_per_grid(self)        


def get_point_counts_per_grid(points,shapes,bbox):
    grid,_ = tools.create_uniform_grid(shapes[0],bbox)
    kdtree = cKDTree(grid)
    _,idx = kdtree.query(points,k=1)
    counts = np.bincount(idx,minlength=len(grid))
    return counts
    
    

def compute_supervoxel_network(segmented_grid, wn_field,points_count, mask: np.ndarray = None):
    """
    构造超体素网络的邻接矩阵
    参数:
      segmented_grid: 分割后的超体素标签网格
      wn_field: 原始的 winding number 场（3D numpy 数组）
    返回:
      G: 超体素网络邻接矩阵，形状为 [K, K]，K 为超体素数量
      G_pointcount 邻接面上点的数量
    """
    if mask is None:
        mask = np.ones_like(segmented_grid,dtype=np.bool)
    labels = np.unique(segmented_grid[mask])
    K = len(labels)
    assert segmented_grid.max() == K-1, "要求segmented_grid的标签从0开始连续排列"
    G = np.zeros((K, K), dtype=np.float32)
    G_pointcount = np.zeros((K, K), dtype=np.float32)
    visited = np.zeros_like(G)
    # 针对每个轴以及正负方向，统计不同超体素间的边权
    for axis in range(3):
        for shift in [1, -1]:
            shifted_sp = np.roll(segmented_grid, shift, axis=axis)
            shifted_wn = np.roll(wn_field, shift, axis=axis)
            shifted_mask = np.roll(mask, shift, axis=axis)

            # 只有没被mask的部分 且shift后也没被mask的 且shift后sp不同的
            tmask = (segmented_grid != shifted_sp) & shifted_mask & mask

            if tmask.sum() == 0:
                print("Warning: No valid connections found between supervoxels in the current shift.")
                continue                

            # 边界处理，避免错误连接
            if shift == 1:
                if axis == 0:
                    tmask[-1, :, :] = False
                elif axis == 1:
                    tmask[:, -1, :] = False
                elif axis == 2:
                    tmask[:, :, -1] = False
            elif shift == -1:
                if axis == 0:
                    tmask[0, :, :] = False
                elif axis == 1:
                    tmask[:, 0, :] = False
                elif axis == 2:
                    tmask[:, :, 0] = False
            else:
                assert False, "shift 只能为 1 或 -1"

            current_labels = segmented_grid[tmask]
            neighbor_labels = shifted_sp[tmask]
            current_point_count = points_count[tmask]
            neighbor_point_count = points_count[tmask]
            
            assert current_labels.min() >= 0
            assert neighbor_labels.min() >= 0
            # 计算winding number的差值的绝对值
            wn_diff = np.abs(wn_field[tmask] - shifted_wn[tmask])

            visited[current_labels, neighbor_labels] = 1
            visited[neighbor_labels, current_labels] = 1
            G[current_labels.flatten(), neighbor_labels.flatten()] += wn_diff.flatten()
            G[neighbor_labels.flatten(), current_labels.flatten()] += wn_diff.flatten() # 保证对称性
            G_pointcount[current_labels.flatten(), neighbor_labels.flatten()] += current_point_count.flatten()
            G_pointcount[neighbor_labels.flatten(), current_labels.flatten()] += neighbor_point_count.flatten()
            
    G[visited==1] += 0.001 # 两个grid，尤其是存在clip操作的情况下，可能边界处边权为零
    nG = nx.from_numpy_array(G)
    components = list(nx.connected_components(nG))
    assert len(components) == 1, "G 的联通分量数量不为 1"
    return G,G_pointcount


def is_connected(G):
    nG = nx.from_numpy_array(G)
    components = list(nx.connected_components(nG))
    return len(components) == 1

# # 3. 使用谱聚类对超体素网络进行二分
# def spectral_bisect_network(G):
#     """
#     利用谱聚类对超体素网络进行二分
#     参数:
#       G: 超体素网络的邻接矩阵（形状为 [K, K]）
#     返回:
#       clusters: 每个超体素的二分标签（0 或 1），长度为 K
#     """
#     spectral = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=0)    
#     sp_G = G.copy()
#     # 非零部分取倒数. TODO 在外部限制值的大小
#     sp_G[G!=0] = 1/G[G!=0]
#     sp_G = np.clip(sp_G,0,15)
#     sp_G = np.exp(sp_G)
#     # sp_G = np.exp(-1 * (sp_G ** 2))
#     clusters = spectral.fit_predict(sp_G)
#     return clusters

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
    # field_data = np.clip(field_data, -1, 1)
    
    # 1. 超体素分割
    segmented_grid,segments_sum,segments_size,segments_mean = segment_winding_field(field_data, n_segments, compactness)
    
    # 2. 构建超体素网络
    G = compute_supervoxel_network(segmented_grid, field_data)
    # G = np.zeros((len(segments_sum),len(segments_sum)))
    # for i in range(len(segments_sum)):
    #     for j in range(len(segments_sum)):
    #         G[i,j] = np.abs(segments_mean[i] - segments_mean[j])
    
    
    # 3. 谱聚类
    clusters = spectral_bisect_network(G)
    
    # 4. 将聚类结果映射回体素网格 to accelerate
    cluster_grid = np.zeros_like(field_data)
    for i in range(len(clusters)):
        cluster_grid[segmented_grid == i] = clusters[i]
    return cluster_grid,segmented_grid,G,clusters





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
    
    cluster_grid,segmented_grid,G,clusters = field_to_clusters(field_data, n_segments, compactness)
    
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


    
import numpy as np
import open3d as o3d

def add_arrow(o3dmesh, start, end, color=(0, 0, 0), radius=0.001):
    # 计算方向向量
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    
    if length == 0:
        print("Start and end points are the same. Cannot create an arrow.")
        return

    direction /= length  # 归一化方向向量

    # 创建箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, 
        cone_radius=2 * radius, 
        cylinder_height=0.8 * length, 
        cone_height=0.2 * length
    )
    color = color[:3]
    arrow.paint_uniform_color(color)  # 赋予颜色
    # 计算旋转矩阵（将默认Z轴方向对齐到direction）
    z_axis = np.array([0, 0, 1])  # Open3D的箭头默认沿Z轴
    rot_axis = np.cross(z_axis, direction)  # 旋转轴
    cos_theta = np.dot(z_axis, direction)  # 旋转角度的余弦值
    sin_theta = np.linalg.norm(rot_axis)

    if sin_theta > 1e-6:  # 避免除零错误
        rot_axis /= sin_theta
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                      [rot_axis[2], 0, -rot_axis[0]],
                      [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)  # 罗德里格斯公式
    else:
        R = np.eye(3) if cos_theta > 0 else -np.eye(3)  # 180度旋转情况

    # 旋转和平移变换
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(start)

    # 添加箭头到 o3dmesh
    o3dmesh += arrow
    return o3dmesh

def add_point(o3dmesh,point,color=(0,0,0),radius=0.001):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    o3dmesh += sphere
    return o3dmesh


import matplotlib.pyplot as plt
def weighted_graph_2_mesh(G,pos,vis=False):
    colormap = plt.cm.viridis
    o3dmesh = o3d.geometry.TriangleMesh()
    for i in range(len(pos)):
        add_point(o3dmesh,pos[i])
    # 获取上三角矩阵中所有非零边的索引
    rows, cols = np.triu_indices(len(pos), k=1)  # 获取上三角矩阵的索引（不包括对角线）
    mask = G[rows, cols] != 0  # 找到非零边
    rows, cols = rows[mask], cols[mask]  # 只保留非零边的索引
    weights = G[rows, cols]  # 获取对应的权重值
    
    # 批量添加箭头
    for start_idx, end_idx, weight in zip(rows, cols, weights):
        add_arrow(o3dmesh, pos[start_idx], pos[end_idx], color=colormap(weight))
    if vis:
        o3d.visualization.draw_geometries([o3dmesh])
    return o3dmesh

    
def get_subgraph(G,ids):
    nid2oid = {}
    for i,id in enumerate(ids):
        nid2oid[i] = id
    nG = np.zeros((len(ids),len(ids)))
    xx,yy = np.meshgrid(range(len(ids)),range(len(ids)))
    xx = xx.flatten()
    yy = yy.flatten()
    get_val = np.vectorize(lambda x: nid2oid[x])
    nG[xx,yy] = G[get_val(xx),get_val(yy)]
    return nG,nid2oid


