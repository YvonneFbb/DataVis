
import cv2
import os
import numpy as np

def preprocess_image(input_path, output_path, alpha=1.5, beta=10):
    """
    对输入的古籍图片进行预处理。

    :param input_path: 输入图片的路径
    :param output_path: 处理后图片的保存路径
    :param alpha: 对比度控制 (1.0-3.0)
    :param beta: 亮度控制 (0-100)
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法读取图片 {input_path}")
        return

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

if __name__ == '__main__':
    # --- 可调整参数 ---
    ALPHA = 1.5  # 对比度 (建议范围: 1.0 ~ 3.0)
    BETA = 10    # 亮度 (建议范围: 0 ~ 100)
    # --------------------

    # 定义输入和输出目录
    INPUT_DIR = 'input'
    OUTPUT_DIR = 'output'
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 构建输入和输出文件路径
    input_file = os.path.join(INPUT_DIR, 'demo.jpg')
    
    # 根据参数生成动态输出文件名
    file_name, file_ext = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(
        OUTPUT_DIR, 
        f"{file_name}_preprocessed_alpha{ALPHA}_beta{BETA}{file_ext}"
    )

    preprocess_image(input_file, output_file, alpha=ALPHA, beta=BETA)
