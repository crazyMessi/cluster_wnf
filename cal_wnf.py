# 使用igl计算WNF
import igl
from igl.copyleft.cgal import fast_winding_number
import numpy as np
import open3d as o3d
import torch
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def normalize_points(points):
    """将点云归一化到单位立方体 [0,1]^3 中"""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    scale = max_coords - min_coords
    normalized = (points - min_coords) / scale
    return normalized, min_coords, scale

def estimate_normals(pcd, neighbors=30):
    """使用PCA估计点云的法向量"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighbors)
    )
    pcd.orient_normals_consistent_tangent_plane(10)
    return np.asarray(pcd.normals)

def create_test_queries(n_points=1000):
    """创建测试用的查询点"""
    return np.random.rand(n_points, 3)

def compute_winding_number_torch(query_points, points, normals, point_areas, epsilon, batch_size):
    """
    使用 PyTorch 计算 winding number，基于 "Fast Winding Numbers for Soups and Clouds" 论文
    
    Args:
        query_points: (N, 3) tensor - 查询点
        points: (M, 3) tensor - 点云点
        normals: (M, 3) tensor - 点云法向量
        point_areas: (M,) tensor - 每个点的面积
        epsilon: float - 防止除零的小量
        batch_size: int - 批处理大小
    
    Returns:
        (N,) tensor - 每个查询点的 winding number
    """
    device = query_points.device
    assert device == points.device == normals.device == point_areas.device
    
    N = query_points.shape[0]  # 查询点数量
    M = points.shape[0]        # 点云点数量

    # 初始化结果
    winding_numbers = torch.zeros(N, device=device)
    
    # 批处理计算
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch_queries = query_points[i:end_idx]
        
        # 计算每个查询点到所有点云点的向量
        query_expanded = batch_queries.unsqueeze(1)  # (batch, 1, 3)
        points_expanded = points.unsqueeze(0)        # (1, M, 3)
        vectors = query_expanded - points_expanded    # (batch, M, 3)
        
        # 计算距离
        distances = torch.norm(vectors, dim=2)       # (batch, M)
        distances_cubed = (distances ** 3).unsqueeze(2)  # (batch, M, 1)
        distances_cubed[distances_cubed<epsilon] = epsilon
        
        # 单位化向量
        vectors = vectors / (distances.unsqueeze(2) + 1e-8)
        
        # 计算点积
        normals_expanded = normals.unsqueeze(0)      # (1, M, 3)
        dot_products = torch.sum(vectors * normals_expanded, dim=2)  # (batch, M)
        
        if point_areas is not None:
            # 使用点面积作为权重
            areas_expanded = point_areas.unsqueeze(0)          # (1, M)
            weights = areas_expanded / (distances_cubed.squeeze(2) + 1e-8)  # (batch, M)
            solid_angles = weights * dot_products / (4.0 * np.pi)
        else:
            # 不使用面积权重
            solid_angles = dot_products / (4.0 * np.pi)
        # 对每个查询点求和
        batch_wn = torch.sum(solid_angles, dim=1)    # (batch,)
        winding_numbers[i:end_idx] = batch_wn
    
    return winding_numbers

def compute_winding_number_torch_api(points, normals, query_points, epsilon=1e-8, batch_size=10000):
    A = compute_points_area_by_knn(points)
    return compute_winding_number_torch(query_points, points, normals, A, epsilon, batch_size)

'''
给定点云 根据领域估算points area、法向量，并返回area、normal、knn_index
'''
def compute_points_area_by_knn(P_np, knn=10):
    """
    计算点云中每个点的近似面积
    
    使用 KNN 找到每个点的邻域，然后：
    1. 计算到邻域点的平均距离向量
    2. 与法向量叉乘得到面积
    
    Args:
        P_np: (N, 3) array - 点云坐标
        knn: int - 近邻点数量
    
    Returns:
        (N,) array - 每个点的估计面积
    """
    n = P_np.shape[0]
    # 使用 scikit-learn 的 NearestNeighbors 计算 knn 邻域
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm='auto').fit(P_np)
    _, I_np = nbrs.kneighbors(P_np)
    # 计算每个点的法向量，方法：对邻域点做 PCA，取最小特征值对应的特征向量
    N_np = np.zeros((n, 3))
    for i in range(n):
        neighbors = P_np[I_np[i]]  # shape (knn, 3)
        mean = np.mean(neighbors, axis=0)
        X = neighbors - mean
        # 协方差矩阵
        C = np.dot(X.T, X) / knn
        eigvals, eigvecs = np.linalg.eigh(C)
        n_p = eigvecs[:, 0]
        # 单位化
        n_p = n_p / np.linalg.norm(n_p)
        N_np[i] = n_p

    distances = P_np[I_np[:, 1:]] - P_np[I_np[:, 0]][:,None,:]
    distances = np.mean(distances, axis=1)
    # 直接用distances叉乘N_np, 得到面积
    A_np = np.cross(distances, N_np)
    # 面积即A_np的模长
    A_np = np.linalg.norm(A_np, axis=1)
    return A_np

class WNF:
    """
    Winding Number Field 类
    用于计算和管理点云的 winding number 场
    """
    def __init__(self, points, epsilon=1e-8):
        """
        构造函数
        
        Args:
            points: (N, 3) array/tensor - 点云坐标
            epsilon: float - 计算 winding number 时防止除零的小量
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 转换输入为 torch tensor
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).float()
        self.points = points.to(self.device)
        
        self.epsilon = epsilon
        self.normals = None
        self.point_areas = None
        
        # 计算点面积
        self._compute_point_areas()
    
    def _compute_point_areas(self, k=10):
        """计算每个点的面积"""
        points_np = self.points.cpu().numpy()
        A = compute_points_area_by_knn(points_np, knn=k)
        self.point_areas = torch.from_numpy(A).float().to(self.device)
    
    def init_normal(self, k=30):
        """
        使用 PCA 初始化法向量
        
        Args:
            k: int - 计算法向量时使用的近邻点数
        """
        points_np = self.points.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        
        # 估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        pcd.orient_normals_consistent_tangent_plane(k)
        
        # 更新法向量
        normals_np = np.asarray(pcd.normals)
        self.normals = torch.from_numpy(normals_np).float().to(self.device)
        
        return self
    
    def update_normal(self, normals):
        """
        更新法向量
        
        Args:
            normals: (N, 3) array/tensor - 新的法向量
        """
        if not isinstance(normals, torch.Tensor):
            normals = torch.from_numpy(normals).float()
        
        assert normals.shape == self.points.shape, "法向量形状必须与点云相同"
        
        self.normals = normals.to(self.device)
        return self
    
    def query_wn(self, query_points, batch_size=10000):
        """
        查询指定位置的 winding number
        
        Args:
            query_points: (M, 3) array/tensor - 查询点坐标
            batch_size: int - 批处理大小
            
        Returns:
            (M,) array/tensor - 查询点的 winding number，
            返回类型与输入类型相同
        """
        assert self.normals is not None, "请先初始化或更新法向量"
        
        fromnp = isinstance(query_points, np.ndarray)
        if fromnp:
            query_points = torch.from_numpy(query_points).float()
        if query_points.device != self.device:
            query_points = query_points.to(self.device)
        
        res = compute_winding_number_torch(
            query_points, 
            self.points, 
            self.normals, 
            self.point_areas,
            self.epsilon,
            batch_size
        )
        
        return res.cpu().numpy() if fromnp else res        
    
    def __call__(self, query_points, batch_size=10000):
        """
        使类实例可调用，等同于 query_wn
        """
        return self.query_wn(query_points, batch_size)

    def igl_wn(self, points, normals, query_points):
        """
        使用 igl 计算 winding number
        """
        return fast_winding_number(self.points, self.normals, query_points)


# 使用示例
if __name__ == "__main__":
    # 1. 创建一个简单的球面点云进行测试
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    # 2. 创建 WNF 对象
    wnf = WNF(points)
    
    # 3. 初始化法向量
    wnf.init_normal()
    
    # 4. 创建测试查询点
    query_points = np.array([
        [0, 0, 0],      # 球心内部点，应该接近 1
        [2, 0, 0],      # 球外点，应该接近 0
        [0.5, 0, 0],    # 球内点，应该接近 1
    ])
    
    # 5. 查询 winding number
    wn = wnf(query_points)
    
    print("测试点的 winding number:")
    print(f"球心点: {wn[0]:.3f} (应接近 1)")
    print(f"球外点: {wn[1]:.3f} (应接近 0)")
    print(f"球内点: {wn[2]:.3f} (应接近 1)")





