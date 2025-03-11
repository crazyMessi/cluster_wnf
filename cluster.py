import numpy as np
from skimage import morphology
import cc3d

def morphological_subdivision_3d(components, kernel, iterations=1):
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
    
    # 当还有未填充的区域时继续膨胀
    while np.any(unfilled_mask):
        # 对每个现有的标签进行一次膨胀
        labels_to_process = np.unique(dilated_components)
        labels_to_process = labels_to_process[labels_to_process != -1]
        
        # 创建临时数组来存储这一轮的膨胀结果
        temp_dilated = dilated_components.copy()
        
        for label in labels_to_process:
            # 获取当前标签的掩码
            current_mask = dilated_components == label
            # 进行一次膨胀
            dilated_mask = morphology.binary_dilation(current_mask, kernel)
            # 限制在原始标签区域内
            valid_dilation = dilated_mask & (original_labels != -1)
            # 只在未填充的区域进行更新
            update_mask = valid_dilation & (temp_dilated == -1)
            temp_dilated[update_mask] = label
        
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
    eroded = morphological_subdivision_3d(components, kernel, iterations)
    
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
