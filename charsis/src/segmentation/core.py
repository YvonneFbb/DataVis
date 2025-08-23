"""
字符分割核心模块 - 重构版本
整合各个子模块的功能，提供统一的接口
"""
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHAR_CLASSIFICATION_CONFIG, VL_CHARACTER_EVALUATION_CONFIG, VL_SEGMENTS_DIR, VL_ANNOTATIONS_DIR
from .detection import analyze_character_dimensions, detect_vertically_connected_chars, detect_wide_chars, reclassify_split_character
from .projection import horizontal_projection_split_vertical_chars, vertical_projection_split_wide_chars
from .merger import merge_narrow_characters
from .visualization import annotate_characters, save_annotated_image, print_classification_stats, create_legend_image
from vl.character_evaluator import VLCharacterEvaluator


def segment_characters(image_path, output_dir=None, min_char_size=None, dilation_iterations=None):
    """
    字符分割主函数 - 重构版本
    
    :param image_path: 输入图像路径
    :param output_dir: 输出目录（可选）
    :param min_char_size: 最小字符尺寸阈值
    :param dilation_iterations: 膨胀操作迭代次数
    :return: 分割后的字符列表和统计信息
    """
    # 从配置中获取默认参数
    if min_char_size is None:
        min_char_size = CHAR_CLASSIFICATION_CONFIG['min_char_size']
    if dilation_iterations is None:
        dilation_iterations = CHAR_CLASSIFICATION_CONFIG['dilation_iterations']
    
    print(f"\n=== 开始处理图像: {os.path.basename(image_path)} ===")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    binary_threshold = CHAR_CLASSIFICATION_CONFIG['binary_threshold']
    _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作
    kernel_size = CHAR_CLASSIFICATION_CONFIG['dilation_kernel_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=dilation_iterations)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小轮廓并获取边界框
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_char_size and h >= min_char_size:
            valid_contours.append((x, y, w, h))
    
    print(f"初始检测到 {len(valid_contours)} 个有效字符")
    
    # 分析字符尺寸
    dimension_stats = analyze_character_dimensions(valid_contours, min_char_size)
    if dimension_stats is None:
        print("未检测到有效字符")
        return [], {'total_merges': 0}
    
    mean_width = dimension_stats['mean_width']
    mean_height = dimension_stats['mean_height']
    std_width = dimension_stats['std_width']
    std_height = dimension_stats['std_height']
    
    # 检测纵向连接字符并分割
    print(f"\n=== 纵向连接字符检测与分割 ===")
    vertically_connected_chars = detect_vertically_connected_chars(valid_contours, mean_height, std_height, mean_width)
    
    # 对纵向连接字符进行水平投影分割
    split_chars = []
    for char in vertically_connected_chars:
        x, y, w, h = char
        split_result = horizontal_projection_split_vertical_chars(gray, x, y, w, h, mean_height)
        split_chars.extend(split_result)
    
    # 更新字符列表
    remaining_chars = [char for char in valid_contours if char not in vertically_connected_chars]
    all_chars = remaining_chars + split_chars
    
    # 检测过宽字符并分割
    print(f"\n=== 过宽字符检测与分割 ===")
    wide_chars = detect_wide_chars(all_chars, mean_width, std_width, mean_height)
    
    # 对过宽字符进行垂直投影分割
    split_wide_chars = []
    for char in wide_chars:
        x, y, w, h = char
        split_result = vertical_projection_split_wide_chars(gray, x, y, w, h, mean_width)
        split_wide_chars.extend(split_result)
    
    # 更新字符列表
    remaining_chars = [char for char in all_chars if char not in wide_chars]
    all_chars = remaining_chars + split_wide_chars
    
    # 重新分析字符尺寸（分割后）
    dimension_stats = analyze_character_dimensions(all_chars, min_char_size)
    if dimension_stats is None:
        print("分割后未检测到有效字符")
        return [], {'total_merges': 0}
    
    mean_width = dimension_stats['mean_width']
    mean_height = dimension_stats['mean_height']
    std_width = dimension_stats['std_width']
    std_height = dimension_stats['std_height']
    
    # 计算窄字符阈值
    narrow_threshold_std_multiplier = CHAR_CLASSIFICATION_CONFIG['narrow_threshold_std_multiplier']
    narrow_width_threshold = mean_width - narrow_threshold_std_multiplier * std_width
    
    print(f"\n=== 字符合并处理 ===")
    print(f"合并前字符数: {len(all_chars)}")
    
    # 执行字符合并
    merged_chars, merge_stats = merge_narrow_characters(
        all_chars, mean_width, mean_height, narrow_width_threshold
    )
    
    print(f"合并后字符数: {len(merged_chars)}")
    
    # VL字符质量评估
    filtered_chars = merged_chars
    vl_stats = {}
    
    if VL_CHARACTER_EVALUATION_CONFIG.get('enabled', False) and output_dir:
        print(f"\n=== VL字符质量评估 ===")
        try:
            # 使用VL专用目录结构
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vl_output_dir = os.path.join(VL_SEGMENTS_DIR, base_name)
            char_images_dir = os.path.join(vl_output_dir, "char_images")
            os.makedirs(char_images_dir, exist_ok=True)
            
            # 同时在VL标注目录创建对应文件夹
            vl_annotation_dir = os.path.join(VL_ANNOTATIONS_DIR, base_name)
            os.makedirs(vl_annotation_dir, exist_ok=True)
            
            char_image_paths = []
            for i, (x, y, w, h, char_type) in enumerate(merged_chars):
                char_img = image[y:y+h, x:x+w]
                char_img_path = os.path.join(char_images_dir, f"char_{i:04d}.png")
                cv2.imwrite(char_img_path, char_img)
                char_image_paths.append(char_img_path)
            
            print(f"已保存 {len(char_image_paths)} 个字符图片用于VL评估")
            
            # 使用VL模型评估字符质量
            evaluator = VLCharacterEvaluator()
            evaluation_results = evaluator.evaluate_batch(char_image_paths)
            
            # 根据评估结果过滤字符
            quality_threshold = VL_CHARACTER_EVALUATION_CONFIG.get('quality_threshold', ['GOOD'])
            filtered_chars = []
            
            for i, char in enumerate(merged_chars):
                if i < len(evaluation_results):
                    quality = evaluation_results[i].get('quality', 'UNCLEAR')
                    if quality in quality_threshold:
                        filtered_chars.append(char)
                else:
                    # 评估失败的字符保留
                    filtered_chars.append(char)
            
            vl_stats = {
                'total_evaluated': len(evaluation_results),
                'good_chars': len([r for r in evaluation_results if r.get('quality') == 'GOOD']),
                'partial_chars': len([r for r in evaluation_results if r.get('quality') == 'PARTIAL']),
                'multiple_chars': len([r for r in evaluation_results if r.get('quality') == 'MULTIPLE']),
                'noise_chars': len([r for r in evaluation_results if r.get('quality') == 'NOISE']),
                'unclear_chars': len([r for r in evaluation_results if r.get('quality') == 'UNCLEAR']),
                'filtered_count': len(merged_chars) - len(filtered_chars),
                'final_count': len(filtered_chars)
            }
            
            print(f"VL评估完成:")
            print(f"  评估字符数: {vl_stats['total_evaluated']}")
            print(f"  优质字符: {vl_stats['good_chars']}")
            print(f"  不完整字符: {vl_stats['partial_chars']}")
            print(f"  多字符: {vl_stats['multiple_chars']}")
            print(f"  噪音字符: {vl_stats['noise_chars']}")
            print(f"  不清楚字符: {vl_stats['unclear_chars']}")
            print(f"  过滤字符数: {vl_stats['filtered_count']}")
            print(f"  最终字符数: {vl_stats['final_count']}")
            
            # 保存VL筛选后的字符图片到VL目录
            print(f"保存VL筛选后的字符图片...")
            filtered_chars_dir = os.path.join(vl_output_dir, "filtered_chars")
            os.makedirs(filtered_chars_dir, exist_ok=True)
            
            for i, (x, y, w, h, char_type) in enumerate(filtered_chars):
                char_img = image[y:y+h, x:x+w]
                filtered_char_path = os.path.join(filtered_chars_dir, f"char_{i:04d}.png")
                cv2.imwrite(filtered_char_path, char_img)
            
            print(f"已保存 {len(filtered_chars)} 个VL筛选后的字符图片到: {filtered_chars_dir}")
            
        except Exception as e:
            print(f"VL字符质量评估失败: {e}")
            print("将使用未过滤的字符结果")
            filtered_chars = merged_chars
    
    # 字符分类和标注
    if output_dir:
        print(f"\n=== 生成标注图像 ===")
        annotated_image, classification_stats = annotate_characters(
            image, filtered_chars, narrow_width_threshold, 
            mean_width, std_width, narrow_threshold_std_multiplier
        )
        
        # 根据是否使用VL评估决定保存位置
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if VL_CHARACTER_EVALUATION_CONFIG.get('enabled', False):
            # VL评估模式：保存到VL标注目录
            annotation_output_dir = os.path.join(VL_ANNOTATIONS_DIR, base_name)
            os.makedirs(annotation_output_dir, exist_ok=True)
            output_path = save_annotated_image(annotated_image, annotation_output_dir, f"{base_name}_vl_annotated")
            legend_path = save_annotated_image(create_legend_image(), annotation_output_dir, "legend")
            print(f"VL标注图像已保存: {output_path}")
            print(f"图例已保存: {legend_path}")
        else:
            # 传统模式：保存到原始输出目录
            output_path = save_annotated_image(annotated_image, output_dir, f"{base_name}_annotated")
            legend_path = save_annotated_image(create_legend_image(), output_dir, "legend")
            print(f"标注图像已保存: {output_path}")
            print(f"图例已保存: {legend_path}")
        
        # 打印统计信息
        print_classification_stats(classification_stats, merge_stats)
    
    # 合并统计信息
    final_stats = merge_stats.copy()
    final_stats.update(vl_stats)
    
    return filtered_chars, final_stats


def process_all_preprocessed_images(input_dir, output_dir):
    """
    批量处理预处理后的图像
    
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    total_stats = {
        'total_images': len(image_files),
        'total_characters': 0,
        'total_merges': 0
    }
    
    for i, filename in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"处理进度: {i}/{len(image_files)} - {filename}")
        print(f"{'='*60}")
        
        input_path = os.path.join(input_dir, filename)
        
        # 为每个图像创建独立的输出目录
        base_name = os.path.splitext(filename)[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        try:
            chars, stats = segment_characters(input_path, image_output_dir)
            total_stats['total_characters'] += len(chars)
            total_stats['total_merges'] += stats.get('total_merges', 0)
            
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {str(e)}")
            continue
    
    # 打印总体统计
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"处理图像数: {total_stats['total_images']}")
    print(f"总字符数: {total_stats['total_characters']}")
    print(f"总合并次数: {total_stats['total_merges']}")
    print(f"平均每图字符数: {total_stats['total_characters'] / total_stats['total_images']:.1f}")


if __name__ == '__main__':
    from config import PREPROCESSED_DIR, SEGMENTS_DIR
    
    # 测试单个图像
    test_image = os.path.join(PREPROCESSED_DIR, "1.png")
    test_output = SEGMENTS_DIR
    
    if os.path.exists(test_image):
        segment_characters(test_image, test_output)
    else:
        print(f"测试图像不存在: {test_image}")
        
        # 批量处理示例
        input_dir = PREPROCESSED_DIR
        output_dir = SEGMENTS_DIR
        
        if os.path.exists(input_dir):
            process_all_preprocessed_images(input_dir, output_dir)
        else:
            print(f"输入目录不存在: {input_dir}")