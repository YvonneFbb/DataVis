"""
投影分析模块 - 改进版本
负责使用投影分析法进行字符分割
集成了高斯平滑、自适应阈值和智能分割点选择
"""
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECTION_CONFIG


def _smooth_projection(projection, sigma=1.0):
    """
    使用高斯滤波平滑投影曲线 - 改进版本
    相比均值滤波，更好地保持边界信息
    
    :param projection: 投影数组
    :param sigma: 高斯核标准差
    :return: 平滑后的投影
    """
    return ndimage.gaussian_filter1d(projection.astype(float), sigma=sigma)


def _adaptive_threshold_calculation(smoothed_projection, percentile_low=25, percentile_high=75):
    """
    自适应阈值计算 - 改进版本
    基于投影的分布特征而非固定比例
    
    :param smoothed_projection: 平滑后的投影
    :param percentile_low: 低分位数
    :param percentile_high: 高分位数
    :return: 自适应阈值
    """
    q1 = np.percentile(smoothed_projection, percentile_low)
    q3 = np.percentile(smoothed_projection, percentile_high)
    # 阈值设为第一四分位数，更适应不同密度文本
    threshold = q1 + 0.2 * (q3 - q1)
    return threshold


def _find_valley_points(smoothed_projection, threshold, min_distance=3):
    """
    寻找投影曲线的谷值点作为分割候选 - 改进版本
    
    :param smoothed_projection: 平滑后的投影
    :param threshold: 投影阈值
    :param min_distance: 最小峰值间距
    :return: 谷值点位置
    """
    # 翻转投影以寻找谷值（变为峰值）
    inverted_projection = -smoothed_projection
    
    # 只考虑低于阈值的区域
    mask = smoothed_projection < threshold
    inverted_projection[~mask] = np.min(inverted_projection)
    
    # 寻找峰值（对应原投影的谷值）
    peaks, _ = find_peaks(
        inverted_projection,
        distance=min_distance,
        prominence=0.1
    )
    
    return peaks


def _score_split_candidates(projection, candidates, mean_size):
    """
    对分割候选点进行评分 - 改进版本
    综合考虑：投影值、位置合理性、分割效果
    
    :param projection: 原始投影
    :param candidates: 候选分割点
    :param mean_size: 平均字符尺寸
    :return: 评分后的分割点列表
    """
    if len(candidates) == 0:
        return []
    
    scored_candidates = []
    
    for pos in candidates:
        # 1. 投影值评分（越小越好）
        projection_score = 1.0 / (projection[pos] + 1e-6)
        
        # 2. 位置合理性评分
        boundary_penalty = 0.0
        total_length = len(projection)
        if pos < total_length * 0.1 or pos > total_length * 0.9:
            boundary_penalty = 0.5
        
        # 3. 局部最小值评分
        local_window = 3
        start_idx = max(0, pos - local_window)
        end_idx = min(len(projection), pos + local_window + 1)
        local_region = projection[start_idx:end_idx]
        
        is_local_min = projection[pos] == np.min(local_region)
        local_min_score = 1.0 if is_local_min else 0.5
        
        # 综合评分
        total_score = projection_score * local_min_score - boundary_penalty
        scored_candidates.append((pos, total_score))
    
    # 按评分排序，返回位置
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [pos for pos, score in scored_candidates]


def _find_split_regions(projection, mean_size, max_splits=None):
    """
    改进的分割区域检测
    
    :param projection: 投影数组
    :param mean_size: 平均字符尺寸
    :param max_splits: 最大分割数
    :return: 优化的分割点列表
    """
    # 1. 高斯平滑
    sigma = max(1.0, mean_size / 20)
    smoothed = _smooth_projection(projection, sigma=sigma)
    
    # 2. 自适应阈值
    threshold = _adaptive_threshold_calculation(smoothed)
    
    # 3. 寻找谷值候选点
    min_distance = max(3, int(mean_size / 4))
    candidates = _find_valley_points(smoothed, threshold, min_distance)
    
    # 4. 候选点评分和筛选
    scored_splits = _score_split_candidates(projection, candidates, mean_size)
    
    # 5. 限制分割数量
    if max_splits and len(scored_splits) > max_splits:
        scored_splits = scored_splits[:max_splits]
    
    # 6. 验证分割合理性
    validated_splits = _validate_splits(scored_splits, len(projection), mean_size)
    
    return sorted(validated_splits)


def _validate_splits(split_points, total_length, mean_size):
    """
    验证分割点的合理性
    
    :param split_points: 分割点列表
    :param total_length: 总长度
    :param mean_size: 平均字符尺寸
    :return: 验证后的分割点
    """
    if not split_points:
        return []
    
    validated = []
    min_segment_size = mean_size * 0.3
    
    prev_pos = 0
    for split_pos in split_points:
        segment_size = split_pos - prev_pos
        if segment_size >= min_segment_size:
            validated.append(split_pos)
            prev_pos = split_pos
    
    # 检查最后一段
    if total_length - prev_pos >= min_segment_size:
        return validated
    else:
        return validated[:-1] if validated else []


def horizontal_projection_split_vertical_chars(binary_img, x, y, w, h, mean_height):
    """
    使用水平投影分析法对纵向连接的字符进行切割 - 改进版本
    
    :param binary_img: 二值化图像
    :param x, y, w, h: 字符边界框
    :param mean_height: 平均字符高度
    :return: 分割后的字符边界框列表
    """
    # 提取字符区域
    char_region = binary_img[y:y+h, x:x+w]
    
    # 计算水平投影
    horizontal_projection = np.sum(char_region, axis=1) // 255
    
    # 使用改进的分割检测
    estimated_chars = max(2, int(h / mean_height + 0.5))
    max_splits = min(estimated_chars - 1, 5)  # 限制最大分割数
    
    split_points = _find_split_regions(
        horizontal_projection, 
        mean_height, 
        max_splits=max_splits
    )
    
    # 生成分割后的边界框
    valid_components = []
    
    if split_points:
        # 有分割点的情况
        prev_y = 0
        min_height = mean_height * PROJECTION_CONFIG['min_segment_height_ratio']
        
        for split_point in split_points:
            segment_height = split_point - prev_y
            if segment_height >= min_height:
                valid_components.append((x, y + prev_y, w, segment_height))
            prev_y = split_point
        
        # 添加最后一段
        final_height = h - prev_y
        if final_height >= min_height:
            valid_components.append((x, y + prev_y, w, final_height))
    
    # 如果没有有效分割，使用智能均匀分割
    if len(valid_components) < 2:
        valid_components = _uniform_split_vertical(x, y, w, h, mean_height)
    
    return valid_components


def vertical_projection_split_wide_chars(binary_img, x, y, w, h, mean_width):
    """
    使用垂直投影分析法对过宽字符进行切割 - 改进版本
    
    :param binary_img: 二值化图像
    :param x, y, w, h: 字符边界框
    :param mean_width: 平均字符宽度
    :return: 分割后的字符边界框列表
    """
    # 提取字符区域
    char_region = binary_img[y:y+h, x:x+w]
    
    # 计算垂直投影
    vertical_projection = np.sum(char_region, axis=0) // 255
    
    # 使用改进的分割检测
    estimated_chars = max(2, int(w / mean_width + 0.5))
    max_splits = min(estimated_chars - 1, 5)
    
    split_points = _find_split_regions(
        vertical_projection, 
        mean_width, 
        max_splits=max_splits
    )
    
    # 生成分割后的边界框
    valid_components = []
    
    if split_points:
        # 有分割点的情况
        prev_x = 0
        min_width = mean_width * PROJECTION_CONFIG['min_segment_width_ratio']
        
        for split_point in split_points:
            segment_width = split_point - prev_x
            if segment_width >= min_width:
                valid_components.append((x + prev_x, y, segment_width, h))
            prev_x = split_point
        
        # 添加最后一段
        final_width = w - prev_x
        if final_width >= min_width:
            valid_components.append((x + prev_x, y, final_width, h))
    
    # 如果没有有效分割，使用智能均匀分割
    if len(valid_components) < 2:
        valid_components = _uniform_split_horizontal(x, y, w, h, mean_width)
    
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