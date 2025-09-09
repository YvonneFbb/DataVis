"""
Post-OCR 模块：对分割后的字符图片进行识别与重命名
"""
import os
import sys
import re
import cv2
import json
from collections import defaultdict
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

# 确保从仓库根导入 src.config
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from src.config import OCR_FILTER_CONFIG, SEGMENTS_DIR, OCR_DIR
except Exception as e:
    raise RuntimeError(
        f"无法导入 src.config（OCR_FILTER_CONFIG/SEGMENTS_DIR/OCR_DIR）。请从仓库根运行或设置 PYTHONPATH。原始错误: {e}"
    )
from src.utils.path import ensure_dir

try:
    # 远程客户端位于 preocr 阶段
    from preocr.remote_paddleocr import RemotePaddleOCRClient
    OCR_AVAILABLE = True
except ImportError:
    print("警告: RemotePaddleOCRClient 未安装或不可用。OCR功能将被禁用。")
    OCR_AVAILABLE = False


def safe_filename(text: str, max_length: int = 50) -> str:
    if not text or not text.strip():
        return "empty"
    safe_text = re.sub(r'[<>:"/\\|?*]', '_', text.strip())
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length]
    return safe_text


def recognize_character_images(input_dir: str, output_dir: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    if not OCR_AVAILABLE:
        print("OCR功能不可用，跳过处理")
        return {'ocr_available': False}
    config = config or OCR_FILTER_CONFIG.copy()
    print(f"\n=== OCR识别开始 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    ensure_dir(output_dir)
    char_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('char_')]
    if not char_files:
        print("未找到字符图片文件")
        return {'total_files': 0}
    char_files.sort()
    print(f"找到 {len(char_files)} 个字符图片文件")
    try:
        ocr_client = RemotePaddleOCRClient()
        print(f"远程PaddleOCR客户端初始化成功")
    except Exception as e:
        print(f"远程PaddleOCR客户端初始化失败: {e}")
        return {'error': str(e)}
    stats = {'total_files': len(char_files), 'recognized': 0, 'empty': 0, 'low_confidence': 0, 'error': 0, 'confidence_threshold': config.get('confidence_threshold', 0.3)}
    name_counter = defaultdict(int)
    ocr_results = []
    print(f"开始处理 {len(char_files)} 个文件...")
    for i, filename in enumerate(char_files, 1):
        input_path = os.path.join(input_dir, filename)
        try:
            from pathlib import Path
            result = ocr_client.predict_single_character(Path(input_path))
            if result['success']:
                recognized_text = result['text']
                confidence = result['confidence']
            else:
                recognized_text = ""; confidence = 0.0
                print(f"OCR识别失败: {filename} - {result.get('message', '未知错误')}")
            ocr_result = {'original_file': filename, 'recognized_text': recognized_text, 'confidence': confidence, 'status': 'success'}
            if not recognized_text:
                output_filename = f"empty_{i:03d}.png"; stats['empty'] += 1; ocr_result['status'] = 'empty'
            elif confidence < config.get('confidence_threshold', 0.3):
                output_filename = f"low_conf_{safe_filename(recognized_text)}_{i:03d}.png"; stats['low_confidence'] += 1; ocr_result['status'] = 'low_confidence'
            else:
                safe_text = safe_filename(recognized_text); name_counter[safe_text] += 1
                output_filename = f"{safe_text}.png" if name_counter[safe_text] == 1 else f"{safe_text}_{name_counter[safe_text]:02d}.png"
                stats['recognized'] += 1; ocr_result['status'] = 'recognized'
            output_path = os.path.join(output_dir, output_filename)
            try:
                img_to_save = cv2.imread(input_path)
                if img_to_save is None:
                    import shutil; shutil.copy2(input_path, output_path)
                else:
                    cv2.imwrite(output_path, img_to_save)
            except Exception:
                import shutil; shutil.copy2(input_path, output_path)
            ocr_result['output_file'] = output_filename; ocr_results.append(ocr_result)
            if i % 50 == 0 or i == len(char_files):
                print(f"已处理: {i}/{len(char_files)} ({i/len(char_files)*100:.1f}%)")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}"); stats['error'] += 1
            ocr_results.append({'original_file': filename, 'error': str(e), 'status': 'error'})
    results_file = os.path.join(output_dir, 'ocr_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'config': config, 'results': ocr_results}, f, ensure_ascii=False, indent=2)
    print(f"\n=== OCR识别完成 ===")
    print(f"总文件数: {stats['total_files']}"); print(f"成功识别: {stats['recognized']}")
    print(f"空识别: {stats['empty']}"); print(f"低置信度: {stats['low_confidence']}"); print(f"错误: {stats['error']}")
    print(f"置信度阈值: {stats['confidence_threshold']}"); print(f"结果保存到: {output_dir}")
    print(f"详细结果: {results_file}")
    return stats


def process_all_segment_results(input_base_dir: str = None, output_base_dir: str = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    if input_base_dir is None: input_base_dir = SEGMENTS_DIR
    if output_base_dir is None: output_base_dir = OCR_DIR
    if not os.path.exists(input_base_dir):
        print(f"输入目录不存在: {input_base_dir}"); return {'error': 'input_dir_not_found'}
    image_dirs = []
    for item in os.listdir(input_base_dir):
        item_path = os.path.join(input_base_dir, item)
        if os.path.isdir(item_path):
            has_char_files = any(f.startswith('char_') and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(item_path))
            if has_char_files: image_dirs.append(item)
    if not image_dirs:
        print(f"在 {input_base_dir} 中未找到包含字符图片的文件夹"); return {'total_dirs': 0}
    print(f"\n=== 批量OCR处理开始 ==="); print(f"输入目录: {input_base_dir}"); print(f"输出目录: {output_base_dir}"); print(f"找到 {len(image_dirs)} 个图片文件夹")
    batch_stats = {'total_dirs': len(image_dirs), 'processed_dirs': 0, 'total_files': 0, 'total_recognized': 0, 'total_empty': 0, 'total_low_confidence': 0, 'total_errors': 0}
    for i, dir_name in enumerate(image_dirs, 1):
        print(f"\n{'='*60}"); print(f"处理进度: {i}/{len(image_dirs)} - {dir_name}"); print(f"{'='*60}")
        input_dir = os.path.join(input_base_dir, dir_name); output_dir = os.path.join(output_base_dir, dir_name)
        try:
            stats = recognize_character_images(input_dir, output_dir, config)
            if 'total_files' in stats:
                batch_stats['processed_dirs'] += 1
                batch_stats['total_files'] += stats.get('total_files', 0)
                batch_stats['total_recognized'] += stats.get('recognized', 0)
                batch_stats['total_empty'] += stats.get('empty', 0)
                batch_stats['total_low_confidence'] += stats.get('low_confidence', 0)
                batch_stats['total_errors'] += stats.get('error', 0)
        except Exception as e:
            print(f"处理目录 {dir_name} 时出错: {e}"); batch_stats['total_errors'] += 1
    print(f"\n{'='*60}"); print(f"=== 批量OCR处理完成 ==="); print(f"{'='*60}")
    print(f"处理目录数: {batch_stats['processed_dirs']}/{batch_stats['total_dirs']}")
    print(f"总文件数: {batch_stats['total_files']}"); print(f"成功识别: {batch_stats['total_recognized']}")
    print(f"空识别: {batch_stats['total_empty']}"); print(f"低置信度: {batch_stats['total_low_confidence']}"); print(f"总错误: {batch_stats['total_errors']}")
    if batch_stats['total_files'] > 0:
        success_rate = batch_stats['total_recognized'] / batch_stats['total_files'] * 100; print(f"识别成功率: {success_rate:.1f}%")
    return batch_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Post-OCR: recognize and rename segmented character images')
    parser.add_argument('--input-dir', help='单个字符图片目录（包含 char_*.png）')
    parser.add_argument('--output-dir', help='输出目录，默认写到 OCR_DIR/<folder>')
    parser.add_argument('--conf-thres', type=float, default=None, help='置信度阈值覆盖（可选）')
    args = parser.parse_args()
    if args.input_dir:
        in_dir = args.input_dir
        if not os.path.isdir(in_dir):
            print(f"输入目录不存在: {in_dir}"); sys.exit(1)
        folder = os.path.basename(os.path.normpath(in_dir))
        out_dir = args.output_dir or os.path.join(OCR_DIR, folder)
        cfg = OCR_FILTER_CONFIG.copy()
        if args.conf_thres is not None: cfg['confidence_threshold'] = float(args.conf_thres)
        stats = recognize_character_images(in_dir, out_dir, cfg)
        print(json.dumps({'single': True, 'stats': stats}, ensure_ascii=False, indent=2))
    else:
        batch_stats = process_all_segment_results()
        if batch_stats.get('total_dirs', 0) == 0:
            print("\n没有找到可处理的segment结果。\n请先运行segment模块生成字符图片，然后再运行OCR处理。")
        elif batch_stats.get('processed_dirs', 0) == batch_stats.get('total_dirs', 0):
            print(f"\n✓ 所有 {batch_stats['total_dirs']} 个目录处理成功！")
        else:
            print(f"\n⚠ 部分目录处理失败：{batch_stats['processed_dirs']}/{batch_stats['total_dirs']} 成功")
