import cv2
import os
import numpy as np
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PREPROCESSED_DIR
from utils.path import ensure_dir, get_output_path

def preprocess_image(input_path, output_path, alpha=1.5, beta=10):
    """
    对输入的古籍图片进行预处理。

    :param input_path: 输入图片的路径
    :param output_path: 处理后图片的保存路径
    :param alpha: 对比度控制 (1.0-3.0)
    :param beta: 亮度控制 (0-100)
    :return: 是否处理成功
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法读取图片 {input_path}")
        return False

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 应用CLAHE增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    local_contrast_enhanced = clahe.apply(gray)

    # 4. 应用线性变换调整全局对比度和亮度
    # new_pixel = alpha * old_pixel + beta
    adjusted = cv2.convertScaleAbs(local_contrast_enhanced, alpha=alpha, beta=beta)

    # 5. 保存处理后的图片
    cv2.imwrite(output_path, adjusted)
    print(f"图片已成功处理并保存至 {output_path}")
    return True


def process_all_raw_images(input_dir=None, output_dir=None, alpha=1.5, beta=10):
    """
    批量处理raw目录下的所有图片。
    
    :param input_dir: 输入目录，默认为RAW_DIR
    :param output_dir: 输出目录，默认为PREPROCESSED_DIR  
    :param alpha: 对比度控制 (1.0-3.0)
    :param beta: 亮度控制 (0-100)
    :return: (处理成功数量, 总数量)
    """
    if input_dir is None:
        input_dir = RAW_DIR
    if output_dir is None:
        output_dir = PREPROCESSED_DIR
        
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图片文件
    image_files = []
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        return 0, 0
        
    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_files.append(filename)
    
    if not image_files:
        print(f"警告：在 {input_dir} 中未找到任何图片文件")
        return 0, 0
    
    print(f"\n=== 开始批量预处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"参数: alpha={alpha}, beta={beta}")
    print("-" * 50)
    
    success_count = 0
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        
        # 生成输出文件名
        file_name, file_ext = os.path.splitext(filename)
        output_filename = f"{file_name}_preprocessed_alpha{alpha}_beta{beta}{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(image_files)}] 处理: {filename}")
        
        if preprocess_image(input_path, output_path, alpha, beta):
            success_count += 1
        else:
            print(f"失败: {filename}")
    
    print("-" * 50)
    print(f"=== 批量预处理完成 ===")
    print(f"成功处理: {success_count}/{len(image_files)} 个文件")
    
    return success_count, len(image_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ancient book images (single image or batch)')
    parser.add_argument('--image', help='单张图片路径，若提供则处理该图片；未提供时默认批处理目录')
    parser.add_argument('--input-dir', help='批处理输入目录，默认使用 RAW_DIR')
    parser.add_argument('--output-dir', help='输出目录，默认使用 PREPROCESSED_DIR')
    parser.add_argument('--alpha', type=float, default=1.5, help='对比度控制 (1.0-3.0)')
    parser.add_argument('--beta', type=float, default=10, help='亮度控制 (0-100)')
    args = parser.parse_args()

    if args.image:
        # 单图处理
        in_path = args.image
        out_dir = args.output_dir or PREPROCESSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(in_path))
        out_name = f"{base}_preprocessed_alpha{args.alpha}_beta{args.beta}{ext}"
        out_path = os.path.join(out_dir, out_name)
        ok = preprocess_image(in_path, out_path, alpha=args.alpha, beta=args.beta)
        if ok:
            print(f"✓ 单图处理完成: {out_path}")
        else:
            print("⚠ 单图处理失败")
    else:
        # 批处理
        success_count, total_count = process_all_raw_images(
            input_dir=args.input_dir or RAW_DIR,
            output_dir=args.output_dir or PREPROCESSED_DIR,
            alpha=args.alpha,
            beta=args.beta,
        )
        if total_count == 0:
            print("\n没有找到可处理的图片文件。")
            print("请确保在 data/raw/ 目录中放置图片文件（支持格式：jpg, jpeg, png, bmp, tiff）")
        elif success_count == total_count:
            print(f"\n✓ 所有 {total_count} 个图片文件处理成功！")
        else:
            print(f"\n⚠ 部分文件处理失败：{success_count}/{total_count} 成功")