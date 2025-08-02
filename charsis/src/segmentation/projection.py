"""
投影分析模块
负责使用投影分析法进行字符分割
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECTION_CONFIG


def _smooth_projection(projection, kernel_size):
    """
    平滑投影曲线，减少噪声影响
    
    :param projection: 投影数组
    :param kernel_size: 平滑核大小
    :return: 平滑后的投影
    """
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(projection.astype(float), kernel, mode='same')


def _find_split_regions(smoothed_projection, threshold, adjacent_tolerance):
    """
    寻找分割区域
    
    :param smoothed_projection: 平滑后的投影
    :param threshold: 投影阈值
    :param adjacent_tolerance: 相邻容差
    :return: 分割点列表
    """
    # 找到所有低于阈值的位置（潜在分割点）
    low_projection_indices = np.where(smoothed_projection < threshold)[0]
    
    split_regions = []
    if len(low_projection_indices) > 0:
        current_region_start = low_projection_indices[0]
        current_region_end = low_projection_indices[0]
        
        for i in range(1, len(low_projection_indices)):
            if low_projection_indices[i] - low_projection_indices[i-1] <= adjacent_tolerance:
                current_region_end = low_projection_indices[i]
            else:
                # 保存当前区域的中点作为分割点
                split_point = (current_region_start + current_region_end) // 2
                split_regions.append(split_point)
                current_region_start = low_projection_indices[i]
                current_region_end = low_projection_indices[i]
        
        # 添加最后一个区域
        split_point = (current_region_start + current_region_end) // 2
        split_regions.append(split_point)
    
    return split_regions


def horizontal_projection_split_vertical_chars(binary_img, x, y, w, h, mean_height):
    """
    使用水平投影分析法对纵向连接的字符进行切割
    
    :param binary_img: 二值化图像
    :param x, y, w, h: 字符边界框
    :param mean_height: 平均字符高度
    :return: 分割后的字符边界框列表
    """
    # 提取字符区域
    char_region = binary_img[y:y+h, x:x+w]
    
    # 1. 计算水平投影（每行的白色像素数量）
    horizontal_projection = np.sum(char_region, axis=1) // 255
    
    # 2. 平滑投影曲线
    smoothed_projection = _smooth_projection(
        horizontal_projection, 
        PROJECTION_CONFIG['smoothing_kernel_size']
    )
    
    # 3. 寻找分割点
    projection_threshold = np.mean(smoothed_projection) * PROJECTION_CONFIG['projection_threshold_ratio']
    split_regions = _find_split_regions(
        smoothed_projection, 
        projection_threshold, 
        PROJECTION_CONFIG['adjacent_tolerance']
    )
    
    # 4. 根据分割点生成字符边界框
    valid_components = []
    
    if len(split_regions) > 0:
        # 有找到分割点的情况
        prev_y = 0
        for split_point in split_regions:
            if split_point - prev_y >= mean_height * PROJECTION_CONFIG['min_segment_height_ratio']:
                abs_x = x
                abs_y = y + prev_y
                seg_w = w
                seg_h = split_point - prev_y
                valid_components.append((abs_x, abs_y, seg_w, seg_h))
            prev_y = split_point
        
        # 添加最后一个字符
        if h - prev_y >= mean_height * PROJECTION_CONFIG['min_segment_height_ratio']:
            abs_x = x
            abs_y = y + prev_y
            seg_w = w
            seg_h = h - prev_y
            valid_components.append((abs_x, abs_y, seg_w, seg_h))
    
    # 5. 如果投影分析失败，使用智能均匀分割
    if len(valid_components) < 2:
        valid_components = _uniform_split_vertical(x, y, w, h, mean_height)
    
    # 6. 按y坐标排序（从上到下）
    valid_components.sort(key=lambda comp: comp[1])
    
    return valid_components


def vertical_projection_split_wide_chars(binary_img, x, y, w, h, mean_width):
    """
    使用垂直投影分析法对过宽字符进行切割
    
    :param binary_img: 二值化图像
    :param x, y, w, h: 字符边界框
    :param mean_width: 平均字符宽度
    :return: 分割后的字符边界框列表
    """
    # 提取字符区域
    char_region = binary_img[y:y+h, x:x+w]
    
    # 1. 计算垂直投影（每列的白色像素数量）
    vertical_projection = np.sum(char_region, axis=0) // 255
    
    # 2. 平滑投影曲线
    smoothed_projection = _smooth_projection(
        vertical_projection, 
        PROJECTION_CONFIG['smoothing_kernel_size']
    )
    
    # 3. 寻找分割点
    projection_threshold = np.mean(smoothed_projection) * PROJECTION_CONFIG['projection_threshold_ratio']
    split_regions = _find_split_regions(
        smoothed_projection, 
        projection_threshold, 
        PROJECTION_CONFIG['adjacent_tolerance']
    )
    
    # 4. 根据分割点生成字符边界框
    valid_components = []
    
    if len(split_regions) > 0:
        # 有找到分割点的情况
        prev_x = 0
        for split_point in split_regions:
            if split_point - prev_x >= mean_width * PROJECTION_CONFIG['min_segment_width_ratio']:
                abs_x = x + prev_x
                abs_y = y
                seg_w = split_point - prev_x
                seg_h = h
                valid_components.append((abs_x, abs_y, seg_w, seg_h))
            prev_x = split_point
        
        # 添加最后一个字符
        if w - prev_x >= mean_width * PROJECTION_CONFIG['min_segment_width_ratio']:
            abs_x = x + prev_x
            abs_y = y
            seg_w = w - prev_x
            seg_h = h
            valid_components.append((abs_x, abs_y, seg_w, seg_h))
    
    # 5. 如果投影分析失败，使用智能均匀分割
    if len(valid_components) < 2:
        valid_components = _uniform_split_horizontal(x, y, w, h, mean_width)
    
    # 6. 按x坐标排序（从左到右）
    valid_components.sort(key=lambda comp: comp[0])
    
    return valid_components


def _uniform_split_vertical(x, y, w, h, mean_height):
    """
    垂直方向的智能均匀分割
    """
    estimated_chars = max(2, int(h / mean_height + 0.5))
    segment_height = h // estimated_chars
    valid_components = []
    
    for i in range(estimated_chars):
        seg_y = y + i * segment_height
        seg_h = segment_height if i < estimated_chars - 1 else h - i * segment_height
        # 确保最小高度
        if seg_h >= mean_height * PROJECTION_CONFIG['min_final_height_ratio']:
            valid_components.append((x, seg_y, w, seg_h))
    
    return valid_components


def _uniform_split_horizontal(x, y, w, h, mean_width):
    """
    水平方向的智能均匀分割
    """
    estimated_chars = max(2, int(w / mean_width + 0.5))
    segment_width = w // estimated_chars
    valid_components = []
    
    for i in range(estimated_chars):
        seg_x = x + i * segment_width
        seg_w = segment_width if i < estimated_chars - 1 else w - i * segment_width
        # 确保最小宽度
        if seg_w >= mean_width * PROJECTION_CONFIG['min_final_height_ratio']:
            valid_components.append((seg_x, y, seg_w, h))
    
    return valid_components