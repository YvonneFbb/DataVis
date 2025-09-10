"""
ocrmac-driven character segmentation (纯新实现，无遗留算法)。

本文件仅保留 ocrmac 分割主流程与接口，无任何旧算法代码或未定义符号。
接口: run_on_image, run_on_ocr_regions
"""
from typing import List, Tuple, Dict, Any, Optional
import cv2
import json
import numpy as np
import os

# Lazy import ocrmac
try:
    from ocrmac import ocrmac as _ocrmac
except ImportError as _e:
    _ocrmac = None
    _OCRMAC_IMPORT_ERROR = str(_e)
else:
    _OCRMAC_IMPORT_ERROR = None

try:
    from src.config import (
        PREOCR_DIR, SEGMENTS_DIR,
    CHAR_CROP_CONFIG,
    )
except Exception as e:
    raise RuntimeError(
    f"无法导入必要配置 (PREOCR_DIR / SEGMENTS_DIR / CHAR_CROP_CONFIG)。原始错误: {e}"
    )

def run_on_image(image_path: str, output_dir: str, expected_text: str | None = None,
                 framework: str = 'livetext', recognition_level: str = 'accurate',
                 language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    if _ocrmac is None:
        return {'success': False, 'error': f'ocrmac 未安装: {_OCRMAC_IMPORT_ERROR}'}
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': f'cannot read: {image_path}'}
    os.makedirs(output_dir, exist_ok=True)
    # 强制 LiveText；不做 Vision 回退
    langs = language_preference or ['zh-Hans']
    try:
        o = _ocrmac.OCR(image_path, framework='livetext', recognition_level=recognition_level, language_preference=langs, detail=True)
        res = o.recognize()
    except Exception as e:
        return {'success': False, 'error': f'LiveText 调用失败: {e}'}
    if not res:
        return {'success': False, 'error': 'LiveText 无检测结果'}
    W, H = int(o.image.width), int(o.image.height)
    boxes = []
    for i, tup in enumerate(res):
        if not isinstance(tup, (list, tuple)) or len(tup) != 3:
            continue
        text, conf, nb = tup
        # Apple Vision 的 normalizedBoundingBox 原点在左下角，需做 y 轴翻转
        x = int(round(nb[0] * W))
        y_top = int(round((1.0 - nb[1] - nb[3]) * H))
        w = int(round(nb[2] * W))
        h = int(round(nb[3] * H))
        # 边界裁切，避免越界与负值
        x = max(0, min(W - 1, x))
        y_top = max(0, min(H - 1, y_top))
        w = max(1, min(W - x, w))
        h = max(1, min(H - y_top, h))
        boxes.append({
            'order': i+1,
            'text': str(text),
            'confidence': float(conf),
            'x': x,
            'y': y_top,
            'w': w,
            'h': h,
            'normalized_bbox': [float(nb[0]), float(nb[1]), float(nb[2]), float(nb[3])]
        })
    # sort top->down
    boxes.sort(key=lambda b: (b['y'], b['x']))
    chars_meta = []
    for b in boxes:
        crop = img[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 245)
        if ys.size > 0:
            y1, y2 = int(ys.min()), int(ys.max())+1
            x1, x2 = int(xs.min()), int(xs.max())+1
            crop = crop[y1:y2, x1:x2]
        fname = f"char_{b['order']:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), crop)
        chars_meta.append({
            'filename': fname,
            'bbox': (b['x'], b['y'], b['w'], b['h']),
            'text_hint': b['text'],
            'confidence': b['confidence'],
            'normalized_bbox': b.get('normalized_bbox')
        })
    # overlay
    overlay = img.copy()
    palette = [(255,0,0),(0,160,255),(0,200,0),(180,0,200),(50,50,255),(255,140,0)]
    for b in boxes:
        c = palette[(b['order']-1)%len(palette)]
        cv2.rectangle(overlay, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), c, 2)
        cv2.putText(overlay, str(b['order']), (b['x']+2, b['y']+b['h']-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    overlay_path = os.path.join(output_dir, 'overlay.png')
    cv2.imwrite(overlay_path, overlay)
    stats = {
        'framework': 'livetext',
    'recognition_level': recognition_level,
        'char_candidates': len(boxes),
        'expected_text_len': len(expected_text) if expected_text else None,
    'image_size': [W, H],
    }
    return {
        'success': True,
        'character_count': len(chars_meta),
        'characters': chars_meta,
        'stats': stats,
        'overlay': overlay_path,
    }

def run_on_ocr_regions(dataset: str | None = None,
                       expected_texts: Dict[str, str] | None = None,
                       framework: str = 'livetext', recognition_level: str = 'accurate',
                       language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    def _find_region_images(base_ocr_dir: str, dataset: str | None = None) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        if dataset:
            datasets = [dataset]
        else:
            try:
                datasets = [d for d in os.listdir(base_ocr_dir) if os.path.isdir(os.path.join(base_ocr_dir, d))]
            except FileNotFoundError:
                datasets = []
        for ds in datasets:
            region_dir = os.path.join(base_ocr_dir, ds, 'region_images')
            if not os.path.isdir(region_dir):
                continue
            try:
                for name in os.listdir(region_dir):
                    if not (name.lower().endswith('.jpg') or name.lower().endswith('.png')):
                        continue
                    if not name.startswith('region_'):
                        continue
                    img_path = os.path.join(region_dir, name)
                    region_name = os.path.splitext(name)[0]
                    results.append((ds, region_name, img_path))
            except FileNotFoundError:
                continue
        results.sort(key=lambda t: (t[0], t[1]))
        return results
    items = _find_region_images(PREOCR_DIR, dataset=dataset)
    processed = []
    errors = []
    for ds, region_name, img_path in items:
        out_dir = os.path.join(SEGMENTS_DIR, ds, region_name)
        os.makedirs(out_dir, exist_ok=True)
        exp_text = None
        if expected_texts:
            exp_text = expected_texts.get(region_name) or expected_texts.get(f"{ds}:{region_name}")
        try:
            res = run_on_image(img_path, out_dir, expected_text=exp_text, framework=framework, recognition_level=recognition_level, language_preference=language_preference)
            with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            processed.append({'dataset': ds, 'region': region_name, 'out_dir': out_dir, 'count': res.get('character_count', 0)})
        except Exception as e:
            errors.append({'dataset': ds, 'region': region_name, 'error': str(e)})
    return {'processed': processed, 'errors': errors}
