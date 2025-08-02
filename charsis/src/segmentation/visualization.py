"""
可视化模块
负责字符分类标注和图片保存
"""
import cv2
import numpy as np
import os


def annotate_characters(image, char_list, narrow_width_threshold, mean_width, std_width, narrow_threshold_std_multiplier):
    """
    在图像上标注字符分类
    
    :param image: 输入图像
    :param char_list: 字符列表
    :param narrow_width_threshold: 窄字符阈值
    :param mean_width: 平均宽度
    :param std_width: 宽度标准差
    :param narrow_threshold_std_multiplier: 窄字符阈值标准差倍数
    :return: 标注后的图像和分类统计
    """
    annotated_image = image.copy()
    
    # 计算较窄字符阈值
    narrow_width_threshold_final = mean_width - narrow_threshold_std_multiplier * std_width
    
    # 统计各类字符数量
    stats = {
        'vertically_connected': 0,
        'wide': 0,
        'merged_red_red': 0,
        'merged_red_green': 0,
        'narrow': 0,
        'normal': 0,
        'total': len(char_list)
    }
    
    for char in char_list:
        x, y, w, h = char[:4]
        char_type = char[4] if len(char) > 4 else 'unknown'
        
        # 根据字符类型选择颜色和标签
        if char_type == 'vertically_connected':
            color = (255, 255, 0)  # 青色
            label = 'V'
            stats['vertically_connected'] += 1
        elif char_type == 'wide':
            color = (255, 0, 0)    # 蓝色
            label = 'W'
            stats['wide'] += 1
        elif char_type == 'merged_red_red':
            color = (0, 165, 255)  # 橙色
            label = 'RR'
            stats['merged_red_red'] += 1
        elif char_type == 'merged_red_green':
            color = (0, 255, 255)  # 黄色
            label = 'RG'
            stats['merged_red_green'] += 1
        elif char_type == 'narrow' or w < narrow_width_threshold_final:
            color = (0, 0, 255)    # 红色
            label = 'N'
            stats['narrow'] += 1
        else:
            color = (0, 255, 0)    # 绿色
            label = 'O'
            stats['normal'] += 1
        
        # 绘制边界框
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
        
        # 添加标签
        cv2.putText(annotated_image, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated_image, stats


def save_annotated_image(image, output_path, filename_prefix="annotated"):
    """
    保存标注后的图像
    
    :param image: 标注后的图像
    :param output_path: 输出路径
    :param filename_prefix: 文件名前缀
    :return: 保存的文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 生成输出文件名
    output_filename = f"{filename_prefix}.png"
    full_output_path = os.path.join(output_path, output_filename)
    
    # 保存图像
    cv2.imwrite(full_output_path, image)
    
    return full_output_path


def print_classification_stats(stats, merge_stats=None):
    """
    打印字符分类统计信息
    
    :param stats: 分类统计
    :param merge_stats: 合并统计（可选）
    """
    print(f"\n=== 字符分类统计 ===")
    print(f"纵向连接字符: {stats['vertically_connected']}")
    print(f"过宽字符: {stats['wide']}")
    print(f"红红合并字符: {stats['merged_red_red']}")
    print(f"红绿合并字符: {stats['merged_red_green']}")
    print(f"较窄字符: {stats['narrow']}")
    print(f"正常字符: {stats['normal']}")
    print(f"总字符数: {stats['total']}")
    
    if merge_stats:
        print(f"\n=== 合并统计 ===")
        print(f"红红合并: {merge_stats['red_red_merges']}次 (共{merge_stats['red_red_rounds']}轮)")
        print(f"红绿合并: {merge_stats['red_green_merges']}次")
        print(f"总合并次数: {merge_stats['total_merges']}次")


def create_legend_image(width=300, height=200):
    """
    创建图例图像
    
    :param width: 图例宽度
    :param height: 图例高度
    :return: 图例图像
    """
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 定义图例项
    legend_items = [
        ("V - 纵向连接", (255, 255, 0)),  # 青色
        ("W - 过宽字符", (255, 0, 0)),    # 蓝色
        ("RR - 红红合并", (0, 165, 255)), # 橙色
        ("RG - 红绿合并", (0, 255, 255)), # 黄色
        ("N - 较窄字符", (0, 0, 255)),    # 红色
        ("O - 正常字符", (0, 255, 0))     # 绿色
    ]
    
    y_start = 20
    y_step = 25
    
    for i, (text, color) in enumerate(legend_items):
        y = y_start + i * y_step
        
        # 绘制颜色方块
        cv2.rectangle(legend, (10, y - 10), (30, y + 10), color, -1)
        cv2.rectangle(legend, (10, y - 10), (30, y + 10), (0, 0, 0), 1)
        
        # 添加文字
        cv2.putText(legend, text, (40, y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return legend