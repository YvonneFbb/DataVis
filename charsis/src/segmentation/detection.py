"""
字符检测模块
负责字符类型检测和尺寸分析
"""
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VERTICAL_CHAR_CONFIG, WIDE_CHAR_CONFIG, RECLASSIFY_CONFIG


def analyze_character_dimensions(bboxes, min_char_size=10):
    """
    分析字符边界框的尺寸统计信息
    
    :param bboxes: 边界框列表 [(x, y, w, h), ...]
    :param min_char_size: 最小字符尺寸
    :return: 统计信息字典
    """
    valid_bboxes = []
    widths = []
    heights = []
    aspect_ratios = []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        if w > min_char_size and h > min_char_size:
            valid_bboxes.append((x, y, w, h))
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(h / w)  # 高宽比
    
    if not widths:
        return None
    
    stats = {
        'valid_bboxes': valid_bboxes,
        'mean_width': np.mean(widths),
        'std_width': np.std(widths),
        'median_width': np.median(widths),
        'mean_height': np.mean(heights),
        'std_height': np.std(heights),
        'median_height': np.median(heights),
        'mean_aspect_ratio': np.mean(aspect_ratios),
        'std_aspect_ratio': np.std(aspect_ratios)
    }
    
    return stats


def detect_vertically_connected_chars(bboxes, mean_height, std_height, mean_width):
    """
    检测纵向连接的字符
    
    :param bboxes: 字符边界框列表 [(x, y, w, h), ...]
    :param mean_height: 平均字符高度
    :param std_height: 字符高度标准差
    :param mean_width: 平均字符宽度
    :return: 纵向连接字符列表
    """
    vertically_connected = []
    
    # 使用配置参数的判断条件
    min_height_for_split = mean_height * VERTICAL_CHAR_CONFIG['height_multiplier']
    min_width_for_valid = mean_width * VERTICAL_CHAR_CONFIG['min_width_ratio']
    aspect_ratio_threshold = VERTICAL_CHAR_CONFIG['aspect_ratio_threshold']
    
    for bbox in bboxes:
        x, y, w, h = bbox
        aspect_ratio = h / w
        
        if (h >= min_height_for_split and 
            aspect_ratio > aspect_ratio_threshold and 
            w >= min_width_for_valid):
            vertically_connected.append(bbox)
    
    return vertically_connected


def detect_wide_chars(bboxes, mean_width, std_width, mean_height):
    """
    检测过宽字符（可能需要水平分割）
    
    :param bboxes: 字符边界框列表 [(x, y, w, h), ...]
    :param mean_width: 平均字符宽度
    :param std_width: 字符宽度标准差
    :param mean_height: 平均字符高度
    :return: 过宽字符列表
    """
    wide_chars = []
    
    # 使用配置参数的判断条件
    min_width_for_split = mean_width * WIDE_CHAR_CONFIG['width_multiplier']
    min_height_for_valid = mean_height * WIDE_CHAR_CONFIG['min_height_ratio']
    max_width_height_ratio = WIDE_CHAR_CONFIG['max_width_height_ratio']
    
    for bbox in bboxes:
        x, y, w, h = bbox
        width_height_ratio = w / h
        
        if (w >= min_width_for_split and 
            h >= min_height_for_valid and 
            width_height_ratio <= max_width_height_ratio):
            wide_chars.append(bbox)
    
    return wide_chars


def reclassify_split_character(w, h, mean_width, mean_height):
    """
    对分割后的字符重新分类
    
    :param w, h: 字符宽度和高度
    :param mean_width, mean_height: 平均宽度和高度
    :return: 字符类型 ('narrow', 'normal', 'wide')
    """
    width_ratio = w / mean_width
    
    # 使用配置参数进行分类
    if width_ratio < RECLASSIFY_CONFIG['narrow_threshold']:
        return 'narrow'
    elif width_ratio > RECLASSIFY_CONFIG['wide_threshold']:
        return 'wide'
    else:
        return 'normal'