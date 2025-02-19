import numpy as np

# 0. 计算指定分辨率的 winding number 场
def normalize_points(points):
    """
    将点云归一化到单位立方体 [0, 1]^3 内
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    scale = max_coords - min_coords
    normalized = (points - min_coords) / scale
    return normalized, min_coords, scale

def visualize_partition(segmented_grid, partition_labels):
    """
    将超体素的二分结果映射回原始体素并可视化
    参数:
      segmented_grid: 原始的超体素标签网格
      partition_labels: 二分后每个超体素的标签(0或1)
    """
    new_grid = np.zeros_like(segmented_grid)
    dict_label = {label: partition_labels[label] for label in np.unique(segmented_grid)}
    f = np.vectorize(lambda x: dict_label[x])
    new_grid = f(segmented_grid)
    return new_grid



# 可视化示例代码
def plot_partition(partition_grid, save_path="./temp/partition.png"):
    """
    可视化二分结果
    参数:
      partition_grid: 二分结果网格
    """
    import matplotlib.pyplot as plt
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 选择中间的切片进行显示
    z_slice = partition_grid.shape[2] // 2
    y_slice = partition_grid.shape[1] // 2
    x_slice = partition_grid.shape[0] // 2
    
    # XY平面
    ax1 = fig.add_subplot(131)
    ax1.imshow(partition_grid[:, :, z_slice], cmap='coolwarm')
    ax1.set_title('XY Plane (z=%d)' % z_slice)
    
    # XZ平面
    ax2 = fig.add_subplot(132)
    ax2.imshow(partition_grid[:, y_slice, :], cmap='coolwarm')
    ax2.set_title('XZ Plane (y=%d)' % y_slice)
    
    # YZ平面
    ax3 = fig.add_subplot(133)
    ax3.imshow(partition_grid[x_slice, :, :], cmap='coolwarm')
    ax3.set_title('YZ Plane (x=%d)' % x_slice)
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.show()
