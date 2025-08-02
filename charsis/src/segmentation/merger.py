"""
字符合并模块
负责红红合并和红绿合并逻辑
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IQR_MULTIPLIER, RED_RED_MERGE_CONFIG, RED_GREEN_MERGE_CONFIG


def calculate_robust_dimensions(bounding_boxes):
    """
    计算去除异常值后的字符尺寸平均值（更准确的典型尺寸）
    使用IQR方法去除异常值
    
    :param bounding_boxes: 边界框列表 [(x, y, w, h), ...]
    :return: 去除异常值后的平均宽度和高度
    """
    widths = []
    heights = []
    
    for x, y, w, h in bounding_boxes:
        widths.append(w)
        heights.append(h)
    
    def remove_outliers(data):
        """使用IQR方法去除异常值"""
        if len(data) < 4:  # 数据太少时直接返回平均值
            return np.mean(data)
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # 定义异常值边界
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        
        # 过滤异常值
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        
        return np.mean(filtered_data) if filtered_data else np.mean(data)
    
    robust_width = remove_outliers(widths)
    robust_height = remove_outliers(heights)
    
    return robust_width, robust_height


def classify_characters(char_list, narrow_width_threshold, mean_width):
    """
    对字符进行初始分类
    
    :param char_list: 字符列表
    :param narrow_width_threshold: 窄字符阈值
    :param mean_width: 平均宽度
    :return: 分类后的字符列表 (narrow_chars, normal_chars, wide_chars)
    """
    narrow_chars = []  # 红色字符
    normal_chars = []  # 绿色字符
    wide_chars = []    # 蓝色字符
    
    for char in char_list:
        x, y, w, h = char[:4]
        char_type = char[4] if len(char) > 4 else 'unknown'
        
        if w < narrow_width_threshold:
            narrow_chars.append((x, y, w, h, 'narrow'))
        elif w > mean_width + RED_RED_MERGE_CONFIG['wide_char_multiplier'] * (mean_width * RED_RED_MERGE_CONFIG['wide_char_ratio']):
            wide_chars.append((x, y, w, h, 'wide'))
        else:
            normal_chars.append((x, y, w, h, 'normal'))
    
    return narrow_chars, normal_chars, wide_chars


def red_red_merge(narrow_chars, ref_width, ref_height):
    """
    执行红红合并（多轮迭代）
    
    :param narrow_chars: 窄字符列表
    :param ref_width: 参考宽度
    :param ref_height: 参考高度
    :return: 合并后的字符列表和统计信息
    """
    merge_stats = {'red_red_merges': 0, 'red_red_rounds': 0}
    round_num = 1
    
    while True:
        merged_in_round = []
        used_indices = set()
        round_merges = 0
        
        for i, (x1, y1, w1, h1, _) in enumerate(narrow_chars):
            if i in used_indices:
                continue
                
            best_merge = None
            best_distance = float('inf')
            
            # 寻找最近的红色字符进行合并
            for j, (x2, y2, w2, h2, _) in enumerate(narrow_chars):
                if j <= i or j in used_indices:
                    continue
                    
                # 计算水平距离（边缘间距离）
                if x1 + w1 <= x2:  # 第一个在左，第二个在右
                    horizontal_distance = x2 - (x1 + w1)
                elif x2 + w2 <= x1:  # 第二个在左，第一个在右
                    horizontal_distance = x1 - (x2 + w2)
                else:  # 有水平重叠
                    horizontal_distance = 0
                
                # 计算垂直重叠
                vertical_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                height_overlap_ratio = vertical_overlap / min(h1, h2) if min(h1, h2) > 0 else 0
                
                # 红红合并条件
                max_distance = ref_width * RED_RED_MERGE_CONFIG['max_horizontal_distance']
                min_overlap = RED_RED_MERGE_CONFIG['min_vertical_overlap_ratio']
                
                if horizontal_distance <= max_distance and height_overlap_ratio > min_overlap:
                    if horizontal_distance < best_distance:
                        best_distance = horizontal_distance
                        best_merge = (j, x2, y2, w2, h2)
            
            if best_merge is not None:
                j, x2, y2, w2, h2 = best_merge
                
                # 计算合并后的边界框
                min_x = min(x1, x2)
                max_x = max(x1 + w1, x2 + w2)
                min_y = min(y1, y2)
                max_y = max(y1 + h1, y2 + h2)
                
                merged_w = max_x - min_x
                merged_h = max_y - min_y
                
                # 检查合并后是否合理
                width_ratio = merged_w / ref_width
                height_ratio = merged_h / ref_height
                
                width_range = RED_RED_MERGE_CONFIG['width_ratio_range']
                height_range = RED_RED_MERGE_CONFIG['height_ratio_range']
                
                if width_range[0] <= width_ratio <= width_range[1] and height_range[0] <= height_ratio <= height_range[1]:
                    # 合并成功
                    merged_in_round.append((min_x, min_y, merged_w, merged_h, 'merged_red_red'))
                    used_indices.add(i)
                    used_indices.add(j)
                    round_merges += 1
                else:
                    # 合并失败，保持原状
                    merged_in_round.append((x1, y1, w1, h1, 'narrow'))
                    used_indices.add(i)
            else:
                # 没有找到合并候选
                merged_in_round.append((x1, y1, w1, h1, 'narrow'))
                used_indices.add(i)
        
        # 添加未被合并的字符
        for i, char in enumerate(narrow_chars):
            if i not in used_indices:
                merged_in_round.append(char)
        
        merge_stats['red_red_merges'] += round_merges
        merge_stats['red_red_rounds'] = round_num
        
        if round_merges == 0 or round_num >= RED_RED_MERGE_CONFIG['max_rounds']:
            break
        
        # 准备下一轮
        narrow_chars = [char for char in merged_in_round if char[4] == 'narrow']
        round_num += 1

    return merged_in_round, merge_stats


def red_green_merge(narrow_chars, normal_chars, ref_width, ref_height, narrow_width_threshold, mean_width):
    """
    执行红绿合并
    
    :param narrow_chars: 窄字符列表
    :param normal_chars: 正常字符列表
    :param ref_width: 参考宽度
    :param ref_height: 参考高度
    :param narrow_width_threshold: 窄字符阈值
    :param mean_width: 平均宽度
    :return: 合并后的字符列表和合并次数
    """
    final_chars = []
    used_narrow = set()
    used_normal = set()
    red_green_merges = 0
    
    for i, (x1, y1, w1, h1, _) in enumerate(narrow_chars):
        if i in used_narrow:
            continue
            
        best_merge = None
        best_distance = float('inf')
        
        # 寻找最近的绿色字符
        for j, (x2, y2, w2, h2, _) in enumerate(normal_chars):
            if j in used_normal:
                continue
                
            # 计算水平距离（边缘间距离）
            if x1 + w1 <= x2:  # 红色在左，绿色在右
                horizontal_distance = x2 - (x1 + w1)
            elif x2 + w2 <= x1:  # 绿色在左，红色在右
                horizontal_distance = x1 - (x2 + w2)
            else:  # 有水平重叠
                horizontal_distance = 0
            
            # 计算垂直重叠
            vertical_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            height_overlap_ratio = vertical_overlap / min(h1, h2) if min(h1, h2) > 0 else 0
            
            # 红绿合并条件
            max_distance = RED_GREEN_MERGE_CONFIG['max_horizontal_distance'] or ref_width * 1.0
            min_overlap = RED_GREEN_MERGE_CONFIG['min_vertical_overlap_ratio']
            
            if horizontal_distance <= max_distance and height_overlap_ratio > min_overlap:
                if horizontal_distance < best_distance:
                    best_distance = horizontal_distance
                    best_merge = (j, x2, y2, w2, h2)
        
        if best_merge is not None:
            j, x2, y2, w2, h2 = best_merge
            
            # 计算合并后的边界框
            min_x = min(x1, x2)
            max_x = max(x1 + w1, x2 + w2)
            min_y = min(y1, y2)
            max_y = max(y1 + h1, y2 + h2)
            
            merged_w = max_x - min_x
            merged_h = max_y - min_y
            
            # 检查合并后是否合理
            width_ratio = merged_w / ref_width
            height_ratio = merged_h / ref_height
            
            width_range = RED_GREEN_MERGE_CONFIG['width_ratio_range']
            height_range = RED_GREEN_MERGE_CONFIG['height_ratio_range']
            
            if width_range[0] <= width_ratio <= width_range[1] and height_range[0] <= height_ratio <= height_range[1]:
                # 合并成功，重新分类
                if merged_w < narrow_width_threshold:
                    new_type = 'narrow'  # 仍然是红色
                elif merged_w > mean_width + RED_GREEN_MERGE_CONFIG['wide_char_multiplier'] * (mean_width * RED_GREEN_MERGE_CONFIG['wide_char_ratio']):
                    new_type = 'wide'    # 变成蓝色
                else:
                    new_type = 'normal'  # 变成绿色
                
                final_chars.append((min_x, min_y, merged_w, merged_h, 'merged_red_green'))
                used_narrow.add(i)
                used_normal.add(j)
                red_green_merges += 1
            else:
                # 合并失败，保持原状
                final_chars.append((x1, y1, w1, h1, 'narrow'))
                used_narrow.add(i)
        else:
            # 没有找到合并候选
            final_chars.append((x1, y1, w1, h1, 'narrow'))
            used_narrow.add(i)
    
    # 添加未被合并的绿色字符
    for i, char in enumerate(normal_chars):
        if i not in used_normal:
            final_chars.append(char)
    
    return final_chars, red_green_merges


def merge_narrow_characters(valid_contours, mean_width, mean_height, narrow_width_threshold):
    """
    多轮迭代合并字符：先进行红红合并（多轮），再进行红绿合并
    
    :param valid_contours: 字符轮廓列表 [(x, y, w, h), ...]
    :param mean_width: 平均字符宽度
    :param mean_height: 平均字符高度
    :param narrow_width_threshold: 较窄字符阈值
    :return: 合并后的字符列表和合并统计信息
    """
    # 计算去除异常值后的尺寸作为更准确的参考
    robust_width, robust_height = calculate_robust_dimensions(valid_contours)
    print(f"  尺寸参考: 原始平均宽度{mean_width:.1f}, 去异常值后{robust_width:.1f}")
    print(f"  尺寸参考: 原始平均高度{mean_height:.1f}, 去异常值后{robust_height:.1f}")
    
    # 使用去除异常值后的尺寸作为主要参考
    ref_width = robust_width
    ref_height = robust_height
    
    # 初始分类字符
    narrow_chars, normal_chars, wide_chars = classify_characters(
        valid_contours, narrow_width_threshold, mean_width
    )
    
    merge_stats = {
        'red_red_merges': 0,
        'red_green_merges': 0,
        'total_merges': 0,
        'red_red_rounds': 0
    }
    
    print(f"  初始分类: 红色{len(narrow_chars)}, 绿色{len(normal_chars)}, 蓝色{len(wide_chars)}")
    
    # 第一阶段：多轮红红合并
    print(f"\n第一阶段：红红合并")
    merged_chars, red_red_stats = red_red_merge(narrow_chars, ref_width, ref_height)
    merge_stats.update(red_red_stats)
    
    print(f"  红红合并: {merge_stats['red_red_merges']}次 (共{merge_stats['red_red_rounds']}轮)")
    
    # 重新分类合并后的字符
    narrow_chars = [char for char in merged_chars if char[4] == 'narrow']
    other_chars = [char for char in merged_chars if char[4] != 'narrow']
    
    # 第二阶段：红绿合并
    print(f"\n第二阶段：红绿合并")
    final_chars, red_green_merges = red_green_merge(
        narrow_chars, normal_chars, ref_width, ref_height, 
        narrow_width_threshold, mean_width
    )
    
    merge_stats['red_green_merges'] = red_green_merges
    merge_stats['total_merges'] = merge_stats['red_red_merges'] + merge_stats['red_green_merges']
    
    # 添加蓝色字符和其他字符
    final_chars.extend(wide_chars)
    final_chars.extend(other_chars)
    
    print(f"  红绿合并: {merge_stats['red_green_merges']}次")
    
    return final_chars, merge_stats