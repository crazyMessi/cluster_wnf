import numpy as np
from skimage import morphology
import cc3d
import optimization
import ncut


    # # 谱聚类
    # assert ncut.is_connected(G), "G is not connected"
    # fsegments = segmented_grid[mask]
    # mgrid = grid[mask]
    # pos = np.array([np.mean(mgrid[fsegments==i],axis=0) for i in range(len(G))])
    # # G,nid2oid = ncut.get_subgraph(G,sp_idx)
    # spectral = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=0) 
    # sp_G = G.copy()
    # sp_G[G_pointcount!=0] = G[G_pointcount!=0] * (G_pointcount[G_pointcount!=0])
    # sp_G = tools.Gaussian(sp_G,sigma=1)
    
    # assert ncut.is_connected(sp_G), "sp_G is not connected"
    # o3dmesh = ncut.weighted_graph_2_mesh(sp_G,pos,vis=False)
    # o3d.io.write_triangle_mesh("temp/iter_{}_topology.ply".format(iteration), o3dmesh)
    # clusters = spectral.fit_predict(sp_G)
    # segmented_grid = np.reshape(segmented_grid,grid_shape)
    # cluster_grid = np.zeros_like(segmented_grid)
    # cluster_grid[~mask.reshape(grid_shape)] = -1
    # for j in range(len(clusters)):
    #     cluster_grid[segmented_grid == sp_idx[j]] = clusters[j]
    # view.view_data_and_mesh(cluster_grid,gt_verts,gt_faces,bbox)
    # G1,_ = ncut.get_subgraph(sp_G,sp_idx[clusters==1])
    # o3dmesh = ncut.vis_weighted_graph(G1,pos[clusters==1])
    # o3d.io.write_triangle_mesh("temp/iter_{}_topology_1.ply".format(iteration), o3dmesh)
    # G2,_ = ncut.get_subgraph(sp_G,sp_idx[clusters==0])
    # o3dmesh = ncut.vis_weighted_graph(G2,pos[clusters==0])
    # o3d.io.write_triangle_mesh("temp/iter_{}_topology_0.ply".format(iteration), o3dmesh)
    

def max_cut(labels,wnf_field,points_count,mask):
    sp_labels = np.unique(labels[mask])
    G,G_pointcount = ncut.compute_supervoxel_network(labels,wnf_field,points_count,mask=mask)
    labels[mask], labels_mean = ncut.reallcate_labels_by_mean_val(labels[mask],wnf_field[mask])
    A,B = optimization.getAB2(labels_mean)
    has_edge = G != 0
    A[~has_edge] = 0
    B[~has_edge] = 0
    A *= 1+G_pointcount
    B /= 1+G_pointcount
    x = optimization.MIQP(A,B,has_edge)
    res_labels = np.zeros_like(labels)
    res_labels[~mask] = -1
    for i in range(len(x)):
        if x[i] == 0:
            continue
        res_labels[labels == sp_labels[i]] = x[i]
    return res_labels
    
    
# def reallocate_labels(labels,connectivity=26):
#     '''
#     重新分配mask内的labels
#     对于-1 保持原样
#     每个联通分量重新分配一个唯一的label
#     '''    
#     components = cc3d.connected_components(labels,connectivity=connectivity)
#     unique_labels = np.unique(components)
#     new_idx = 0
#     for label in unique_labels:
#         if label == -1:
#             continue
#         labels[components == label] = new_idx
#         new_idx += 1
#     return labels

def controlled_erosion_3d(components, kernel, iterations=1):
    """
    对3D标签图像进行形态学细分，保持原有边界的同时对各个区域进行过分割
    
    参数:
    components: 输入的3D标签图像
    kernel: 结构元素，用于定义连通性
    iterations: 腐蚀操作的迭代次数
    
    返回:
    new_labels: 过分割后的标签图像
    """
    # 获取唯一的标签值
    unique_labels = np.unique(components)
    # 排除背景标签-1
    unique_labels = unique_labels[unique_labels != -1]
    
    # 创建新的标签图像，初始化为背景(-1)
    eroded_components = np.zeros_like(components) - 1
    new_idx = 1
    
    # 处理每个标签
    for label in unique_labels:
        # 创建当前标签的掩码
        component_mask = components == label
        # 转换为uint8类型进行二值形态学操作
        temp_component = np.zeros_like(components, dtype=np.uint8)
        temp_component[component_mask] = 1
        
        # 进行迭代腐蚀
        for i in range(iterations):
            # 腐蚀操作
            temp_component = morphology.binary_erosion(temp_component, kernel)
            # 找到腐蚀后的连通区域
            temp_component = cc3d.connected_components(temp_component)
            
            # 为每个新的连通区域分配标签
            region_labels = np.unique(temp_component)
            for region_label in region_labels:
                if region_label == 0:  # 跳过背景
                    continue
                # 确保新区域在原始区域内
                region_mask = (temp_component == region_label) & component_mask
                eroded_components[region_mask] = new_idx
                new_idx += 1
    
    return eroded_components

def controlled_dilation_3d(components, original_labels, kernel):
    """
    在原始标签区域内进行受控膨胀，直到填充完所有非背景区域
    
    参数:
    components: 需要膨胀的标签图像
    original_labels: 原始标签图像，用于限制膨胀范围
    kernel: 结构元素，用于定义连通性
    
    返回:
    dilated_components: 膨胀后的标签图像
    """
    # 创建输出数组
    dilated_components = components.copy()
    
    # 获取需要填充的区域（原始非背景区域中还未被标记的部分）
    unfilled_mask = (original_labels != -1) & (dilated_components == -1)
    
    
    # 对每个现有的标签进行一次膨胀
    labels_to_process = np.unique(dilated_components)
    labels_to_process = labels_to_process[labels_to_process != -1]
    
    
    # 当还有未填充的区域时继续膨胀
    while np.any(unfilled_mask):
    # 创建临时数组来存储这一轮的膨胀结果
        temp_dilated = dilated_components.copy()
    
        for label in labels_to_process:
            # 获取当前标签的掩码
            current_mask = dilated_components == label
            # 进行一次膨胀
            dilated_mask = morphology.binary_dilation(current_mask, kernel)
            # 限制在原始标签区域内 且temp_dilated中为-1的区域
            valid_dilation = dilated_mask & (original_labels != -1) & (temp_dilated == -1)
            temp_dilated[valid_dilation] = label
        
        # 更新结果
        dilated_components = temp_dilated
        # 更新未填充掩码
        unfilled_mask = (original_labels != -1) & (dilated_components == -1)
    return dilated_components

def morphological_subdivision_with_dilation_3d(components, kernel, iterations=1):
    """
    完整的形态学处理过程：先腐蚀分割，再控制膨胀，最后确保每个连通分量有唯一标签
    
    参数:
    components: 输入的3D标签图像
    kernel: 结构元素，用于定义连通性
    iterations: 腐蚀操作的迭代次数
    
    返回:
    final_components: 最终的标签图像，每个连通分量都有唯一的标签
    """
    # 先进行腐蚀和分割
    eroded = controlled_erosion_3d(components, kernel, iterations)
    
    # 然后在原始区域内进行受控膨胀
    dilated = controlled_dilation_3d(eroded, components, kernel)
    
    # 最后确保每个连通分量都有唯一的标签
    final_components = np.zeros_like(components) - 1
    new_idx = 1 
    
    # 获取所有非背景标签
    unique_labels = np.unique(dilated)
    unique_labels = unique_labels[unique_labels != -1]
    
    # 处理每个标签
    for label in unique_labels:
        # 获取当前标签的掩码
        mask = dilated == label
        # 找到这个掩码中的所有连通分量
        connected_regions = cc3d.connected_components(mask.astype(np.uint8))
        # 为每个连通分量分配新的唯一标签
        region_labels = np.unique(connected_regions)
        for region_label in region_labels:
            if region_label == 0:  # 跳过背景
                continue
            region_mask = connected_regions == region_label
            final_components[region_mask] = new_idx
            new_idx += 1
    
    return final_components
