import cv2
import os
import numpy as np
import sys
import argparse
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from src.config import RAW_DIR, PREPROCESSED_DIR, PREPROCESS_STROKE_HEAL_CONFIG, PREPROCESS_INK_PRESERVE_CONFIG
except Exception as e:
    raise RuntimeError(
        f"无法导入 src.config（RAW_DIR/PREPROCESSED_DIR 等）。请从仓库根运行或设置 PYTHONPATH。原始错误: {e}"
    )
from src.utils.path import ensure_dir, get_output_path

def _keep_large_components(mask: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Keep only connected components with area >= min_area in a binary mask."""
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[labels == i] = 1
    return (out * 255).astype(np.uint8)

def _build_kernel(size: int, mode: str) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if mode == 'iso':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if mode == 'h':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    if mode == 'v':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    if mode == 'd1':  # main diagonal ↘
        k = np.zeros((size, size), np.uint8)
        np.fill_diagonal(k, 1)
        return k
    if mode == 'd2':  # anti-diagonal ↙
        k = np.zeros((size, size), np.uint8)
        np.fill_diagonal(np.fliplr(k), 1)
        return k
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def _stroke_heal(gray: np.ndarray) -> np.ndarray:
    cfg = PREPROCESS_STROKE_HEAL_CONFIG
    if not cfg.get('enabled', False):
        return gray
    g = gray.copy()
    # 可选双边滤波：去纸纹，保边缘
    if cfg.get('bilateral_denoise', False):
        b = cfg.get('bilateral', {})
        g = cv2.bilateralFilter(g, int(b.get('d', 5)), float(b.get('sigmaColor', 30)), float(b.get('sigmaSpace', 15)))
    # 逆向闭运算：先取反使墨迹为白，再用闭运算填缝
    inv = 255 - g
    size = int(cfg.get('kernel', 3))
    iters = int(cfg.get('iterations', 1))
    dirs = cfg.get('directions', ['iso'])
    healed = inv
    for d in dirs:
        k = _build_kernel(size, d)
        healed = cv2.morphologyEx(healed, cv2.MORPH_CLOSE, k, iterations=iters)
    out = 255 - healed
    return out

def preprocess_image(input_path, output_path, alpha=1.5, beta=10, heal: bool | None = None, heal_debug: bool = False,
                     heal_kernel: int | None = None, heal_iterations: int | None = None,
                     heal_directions: str | None = None, heal_bilateral: bool | None = None):
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

    # 2.5 可选：断笔/小缺口修补（在增强前做）
    use_heal = PREPROCESS_STROKE_HEAL_CONFIG.get('enabled', False) if heal is None else bool(heal)
    if use_heal:
        # 允许本次调用临时覆盖 heal 配置
        if heal_kernel is not None:
            PREPROCESS_STROKE_HEAL_CONFIG['kernel'] = int(heal_kernel)
        if heal_iterations is not None:
            PREPROCESS_STROKE_HEAL_CONFIG['iterations'] = int(heal_iterations)
        if heal_directions:
            PREPROCESS_STROKE_HEAL_CONFIG['directions'] = [d.strip() for d in heal_directions.split(',') if d.strip()]
        if heal_bilateral is not None:
            PREPROCESS_STROKE_HEAL_CONFIG['bilateral_denoise'] = bool(heal_bilateral)
        gray = _stroke_heal(gray)

    # 3. 应用CLAHE增强局部对比度（为便于调试，保留 baseline 与 healed 两路）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # baseline: 无修补
    base_enh = clahe.apply(gray)
    base_adj = cv2.convertScaleAbs(base_enh, alpha=alpha, beta=beta)
    # healed 路由：若未启用修补，等同于 baseline
    healed_gray = gray
    if use_heal:
        healed_gray = gray
        healed_gray = _stroke_heal(healed_gray)
    heal_enh = clahe.apply(healed_gray)
    adjusted = cv2.convertScaleAbs(heal_enh, alpha=alpha, beta=beta)

    # 3.5 墨色保持/增强：黑帽回墨 + 可选反锐化，避免整体发灰
    try:
        ink_cfg = PREPROCESS_INK_PRESERVE_CONFIG
        if ink_cfg.get('enabled', True):
            ksize = int(ink_cfg.get('blackhat_kernel', 9))
            if ksize % 2 == 0:
                ksize += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            # 黑帽提取“暗笔画相对亮背景”的分量
            blackhat = cv2.morphologyEx(adjusted, cv2.MORPH_BLACKHAT, kernel)
            strength = float(ink_cfg.get('blackhat_strength', 0.6))
            # 回墨：把黑帽分量按比例减回去，使笔画更黑
            adjusted = cv2.subtract(adjusted, cv2.convertScaleAbs(blackhat, alpha=strength, beta=0))
            # 可选反锐化（unsharp masking）
            amount = float(ink_cfg.get('unsharp_amount', 0.0))
            if amount > 1e-6:
                blur = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=1.0)
                adjusted = cv2.addWeighted(adjusted, 1 + amount, blur, -amount, 0)
    except Exception as _:
        pass

    # 5. 保存处理后的图片
    cv2.imwrite(output_path, adjusted)
    print(f"图片已成功处理并保存至 {output_path}")
    # 可选调试输出：对比修补前后效果（仅在启用修补时输出）
    if use_heal and heal_debug:
        try:
            # 输出目录与基名
            out_dir = os.path.dirname(output_path)
            base_name, ext = os.path.splitext(os.path.basename(output_path))
            # 仅输出两张调试图：diff 热力图 + 新增增补叠加（第三张为主输出 heal 图）
            absdiff = cv2.absdiff(base_adj, adjusted)
            _, b0 = cv2.threshold(base_adj, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, b1 = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            added = cv2.bitwise_and(cv2.bitwise_not(b0), b1)
            # 过滤小连通域，避免噪声点
            added = _keep_large_components(added, min_area=6)
            # 新增增补叠加（仅画新增，绿色细线，底图为 healed）
            added_overlay = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours((added>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(added_overlay, contours, -1, (0,255,0), thickness=1)
            cv2.imwrite(os.path.join(out_dir, f"{base_name}_added_overlay{ext}"), added_overlay)
            # 差异热力图
            absnorm = cv2.normalize(absdiff, None, 0, 255, cv2.NORM_MINMAX)
            heat = cv2.applyColorMap(absnorm, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out_dir, f"{base_name}_diff_heat{ext}"), heat)
        except Exception as e:
            print(f"调试输出失败: {e}")
    return True


def process_all_raw_images(input_dir=None, output_dir=None, alpha=1.5, beta=10, heal: bool | None = None):
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
    print(f"参数: alpha={alpha}, beta={beta}, heal={PREPROCESS_STROKE_HEAL_CONFIG.get('enabled', False) if heal is None else bool(heal)}")
    print("-" * 50)
    
    success_count = 0
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)

        # 生成输出文件名
        file_name, file_ext = os.path.splitext(filename)
        suffix_heal = "_heal" if (PREPROCESS_STROKE_HEAL_CONFIG.get('enabled', False) if heal is None else bool(heal)) else ""
        output_filename = f"{file_name}_preprocessed{suffix_heal}_alpha{alpha}_beta{beta}{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(image_files)}] 处理: {filename}")

        if preprocess_image(input_path, output_path, alpha, beta, heal=heal):
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
    parser.add_argument('--heal', action='store_true', help='启用断笔/小缺口修补')
    parser.add_argument('--heal-debug', action='store_true', help='保存修补前后对比可视化（输出差异热力与新增增补叠加）')
    parser.add_argument('--heal-kernel', type=int, default=None, help='修补核尺寸（奇数，如 3/5）')
    parser.add_argument('--heal-iterations', type=int, default=None, help='闭运算迭代次数（建议 1-2）')
    parser.add_argument('--heal-directions', type=str, default=None, help='方向列表，如 "iso,h,v" 或 "iso,h,v,d1,d2"')
    parser.add_argument('--heal-bilateral', action='store_true', help='启用修补前的双边滤波去纹理')
    args = parser.parse_args()

    if args.image:
        # 单图处理
        in_path = args.image
        out_dir = args.output_dir or PREPROCESSED_DIR
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(in_path))
        out_name = f"{base}_preprocessed{'_heal' if args.heal or PREPROCESS_STROKE_HEAL_CONFIG.get('enabled', False) else ''}_alpha{args.alpha}_beta{args.beta}{ext}"
        out_path = os.path.join(out_dir, out_name)
        ok = preprocess_image(
            in_path, out_path,
            alpha=args.alpha, beta=args.beta,
            heal=(args.heal or None), heal_debug=args.heal_debug,
            heal_kernel=args.heal_kernel, heal_iterations=args.heal_iterations,
            heal_directions=args.heal_directions, heal_bilateral=(True if args.heal_bilateral else None)
        )
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
            heal=(args.heal or None),
        )
        if total_count == 0:
            print("\n没有找到可处理的图片文件。")
            print("请确保在 data/raw/ 目录中放置图片文件（支持格式：jpg, jpeg, png, bmp, tiff）")
        elif success_count == total_count:
            print(f"\n✓ 所有 {total_count} 个图片文件处理成功！")
        else:
            print(f"\n⚠ 部分文件处理失败：{success_count}/{total_count} 成功")