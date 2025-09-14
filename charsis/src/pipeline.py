#!/usr/bin/env python3
"""
简化管线脚本：自动识别输入(文件或目录)，按阶段串行执行。

阶段关键字 (按默认顺序): preprocess -> preocr -> segment -> ocr

用法示例:
1) 对单张原始图跑全流程 (所有阶段):
   python -m src.pipeline input.jpg
2) 仅做预处理 + 分割 (跳过 preocr / ocr):
   python -m src.pipeline input.jpg --stages preprocess,segment
3) 对一个目录批量处理 (目录中所有支持的图片):
   python -m src.pipeline data/raw --stages preprocess,segment,ocr
4) 已有 preOCR 结果 (有 region_images 子目录)，直接分割 + OCR:
   python -m src.pipeline data/results/preocr/demo --stages segment,ocr

所有参数依赖 config.py，不再暴露细粒度 CLI 选项；如需高级调参使用 pipeline_advanced.py。
"""
from __future__ import annotations
import os, sys, json, glob, shutil
from typing import List, Dict, Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src import config
    from src.config import (
        RAW_DIR, PREPROCESSED_DIR, PREOCR_DIR, SEGMENTS_DIR, POSTOCR_DIR
    )
except Exception as e:
    raise RuntimeError(f"配置导入失败: {e}")

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def _is_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path.lower())[1] in IMG_EXT


def _list_images_in_dir(folder: str) -> List[str]:
    """递归遍历目录查找所有支持的图片文件"""
    files = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in IMG_EXT):
                files.append(os.path.join(root, filename))
    files.sort()
    return files


def stage_preprocess(inputs: List[str], input_base_dir: str) -> List[str]:
    from src.preprocess.core import preprocess_image
    outputs = []
    for p in inputs:
        # 计算相对路径以保持目录结构
        rel_path = os.path.relpath(p, input_base_dir)
        rel_dir = os.path.dirname(rel_path)
        base, ext = os.path.splitext(os.path.basename(p))

        # 创建对应的输出目录结构
        if rel_dir and rel_dir != '.':
            out_dir = os.path.join(PREPROCESSED_DIR, rel_dir)
        else:
            out_dir = PREPROCESSED_DIR
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{base}_preprocessed{ext}")
        ok = preprocess_image(p, out_path)
        if ok:
            outputs.append(out_path)
    return outputs


def stage_preocr(inputs: List[str], input_base_dir: str) -> Dict[str, Any]:
    # 使用远程服务对每张图做区域检测
    from src.preocr.run_preocr import run_preocr_on_single_image
    summary: Dict[str, Any] = {'inputs': len(inputs), 'details': []}
    for p in inputs:
        try:
            # 计算相对路径以保持目录结构
            rel_path = os.path.relpath(p, input_base_dir)
            rel_dir = os.path.dirname(rel_path)
            r = run_preocr_on_single_image(p, rel_dir)
        except Exception as e:
            r = {'error': str(e), 'input': p}
        summary['details'].append(r)
    return summary


def _collect_region_images(preocr_root: str) -> List[str]:
    region_imgs: List[str] = []
    if os.path.isdir(preocr_root):
        # 常见结构: <preocr_root>/region_images/*.jpg
        region_dir = os.path.join(preocr_root, 'region_images')
        if os.path.isdir(region_dir):
            for ext in IMG_EXT:
                region_imgs.extend(glob.glob(os.path.join(region_dir, f'*{ext}')))
        # 或 dataset 层级: preocr/<dataset>/region_images
        else:
            for ds in os.listdir(preocr_root):
                dpath = os.path.join(preocr_root, ds, 'region_images')
                if os.path.isdir(dpath):
                    for ext in IMG_EXT:
                        region_imgs.extend(glob.glob(os.path.join(dpath, f'*{ext}')))
    region_imgs.sort()
    return region_imgs


def stage_segment(region_images: List[str]) -> Dict[str, Any]:
    from src.segmentation.vertical_hybrid import run_on_image
    processed = []
    for img_path in region_images:
        # out dir: segments/<dataset_or_misc>/<region_name>
        # 解析 dataset: 如果路径包含 /preocr/<ds>/region_images/region_xxx.jpg
        parts = img_path.split(os.sep)
        try:
            idx = parts.index('preocr')
            dataset = parts[idx+1]
        except ValueError:
            dataset = 'misc'
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(SEGMENTS_DIR, dataset, base)
        os.makedirs(out_dir, exist_ok=True)
        res = run_on_image(img_path, out_dir, framework='livetext')
        overlay_path = res.get('overlay')
        overlay_target = None
        if overlay_path and os.path.isfile(overlay_path):
            overlay_dir = os.path.join(SEGMENTS_DIR, dataset, '_overlays')
            os.makedirs(overlay_dir, exist_ok=True)
            overlay_target = os.path.join(overlay_dir, f"{base}_overlay.png")
            try:
                shutil.copy2(overlay_path, overlay_target)
            except Exception as e:
                print(f"warning: overlay copy failed for {overlay_path}: {e}")
                overlay_target = None
        processed.append({
            'image': img_path,
            'out_dir': out_dir,
            'count': res.get('character_count', 0),
            'overlay': overlay_path,
            'overlay_aggregated': overlay_target,
        })
    return {'processed': processed}


def stage_postocr() -> Dict[str, Any]:
    from src.postocr.core import process_all_segment_results
    force = bool(os.getenv('POSTOCR_FORCE'))
    return process_all_segment_results(config={'force': force})


def run_pipeline(input_path: str, stages: List[str]) -> Dict[str, Any]:
    config.validate_config()
    result: Dict[str, Any] = {'input': input_path, 'stages': stages}
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'输入不存在: {input_path}')

    # 1. 收集初始图片列表 (raw set)
    if os.path.isdir(input_path):
        raw_images = _list_images_in_dir(input_path)
    elif _is_image(input_path):
        raw_images = [input_path]
    else:
        raise ValueError('输入既不是图片也不是目录')
    result['raw_count'] = len(raw_images)

    current_images = raw_images
    region_images: List[str] = []

    for st in stages:
        if st == 'preprocess':
            current_images = stage_preprocess(current_images, input_path)
            result['preprocess'] = {'count': len(current_images)}
        elif st == 'preocr':
            # preocr 阶段使用 preprocessed 目录作为基准
            preocr_info = stage_preocr(current_images, PREPROCESSED_DIR)
            result['preocr'] = preocr_info
            # 尝试收集 region 图（如果用户已事先跑过 preOCR）
            region_images = _collect_region_images(PREOCR_DIR)
            result['region_images_found'] = len(region_images)
        elif st == 'segment':
            if not region_images:
                # 如果用户直接传入了 preocr 数据集目录，则从该目录收集 region_images
                if os.path.isdir(input_path) and ('preocr' in os.path.normpath(input_path).split(os.sep)):
                    region_images = _collect_region_images(input_path)
                # 否则退回到当前图片列表
                if not region_images:
                    region_images = current_images
            seg_info = stage_segment(region_images)
            result['segment'] = {'processed': len(seg_info.get('processed', []))}
        elif st in ('ocr', 'postocr'):
            post_stats = stage_postocr()
            result['postocr'] = post_stats
        else:
            raise ValueError(f'未知阶段: {st}')

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='简化流水线：根据阶段执行（配置来自 config.py）')
    parser.add_argument('input', help='输入文件或目录')
    parser.add_argument('--stages', default='preprocess,segment,postocr', help='逗号分隔阶段序列: preprocess,preocr,segment,postocr')
    parser.add_argument('--stage', default=None, help='单阶段快捷参数（等价于 --stages 单值），例如: --stage segment')
    parser.add_argument('--json', action='store_true', help='输出 JSON 结果')
    args = parser.parse_args()

    if args.stage:
        stages = [s.strip() for s in args.stage.split(',') if s.strip()]
    else:
        stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    try:
        out = run_pipeline(args.input, stages)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"完成: stages={stages} raw={out.get('raw_count')}")
    except Exception as e:
        print(f"失败: {e}")
        if os.environ.get('PIPELINE_RAISE'):
            raise
