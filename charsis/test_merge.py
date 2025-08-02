#!/usr/bin/env python3
"""
测试多轮合并策略
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from segmentation.core import segment_characters
from preprocess.core import preprocess_image
import cv2

def main():
    # 设置路径
    project_root = os.path.dirname(__file__)
    input_path = os.path.join(project_root, 'data', 'raw', 'demo.jpg')
    output_dir = os.path.join(project_root, 'data', 'results')
    processed_path = os.path.join(project_root, 'data', 'results', 'processed_demo.jpg')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入图像: {input_path}")
    print(f"输出目录: {output_dir}")
    
    # 预处理
    preprocess_image(input_path, processed_path)
    
    # 字符分割
    merged_chars, merge_stats = segment_characters(processed_path, output_dir)
    
    if merged_chars is None:
        print("字符分割失败")
        return
    
    print(f"\n=== 处理完成 ===")
    print(f"最终字符数: {len(merged_chars)} 个")
    
    if merge_stats:
        print(f"红红合并: {merge_stats.get('red_red_merges', 0)} 次")
        print(f"红绿合并: {merge_stats.get('red_green_merges', 0)} 次")
        print(f"总合并次数: {merge_stats.get('total_merges', 0)} 次")
    
    print(f"\n结果图像已保存到目录: {output_dir}")

if __name__ == "__main__":
    main()