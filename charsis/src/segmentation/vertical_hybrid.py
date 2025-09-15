"""
ocrmac-driven segmentation (border-first, simplified).

LiveText detection → border trimming (src/segmentation/border_trim.py)
→ save crops and debug. Legacy projection/grabcut/morph code moved to
vertical_hybrid_legacy.py and is not used by default.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import cv2
import numpy as np

# Lazy import ocrmac (Apple LiveText)
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
        SEGMENT_REFINE_CONFIG, BORDER_TRIM_CONFIG,
    )
except Exception as e:
    raise RuntimeError(
        f"无法导入必要配置 (PREOCR_DIR / SEGMENTS_DIR / SEGMENT_REFINE_CONFIG / BORDER_TRIM_CONFIG)。原始错误: {e}"
    )

from src.segmentation.border_trim import trim_borders, binarize


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _render_border_debug(roi_bgr: np.ndarray, xl: int, xr: int, yt: int, yb: int,
                         bin_mode: str = 'otsu', ablock: int = 31, aC: int = 3) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    vis = roi_bgr.copy()
    # draw cuts
    cv2.line(vis, (max(0, int(xl)), 0), (max(0, int(xl)), h-1), (0,0,255), 1)
    cv2.line(vis, (max(0, int(xr-1)), 0), (max(0, int(xr-1)), h-1), (0,0,255), 1)
    cv2.line(vis, (0, max(0, int(yt))), (w-1, max(0, int(yt))), (0,0,255), 1)
    cv2.line(vis, (0, max(0, int(yb-1))), (w-1, max(0, int(yb-1))), (0,0,255), 1)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    bin_img = binarize(gray, mode=bin_mode, adaptive_block=ablock, adaptive_C=aC)
    vproj = (bin_img > 0).sum(axis=0).astype(np.float32)
    hproj = (bin_img > 0).sum(axis=1).astype(np.float32)
    # vertical panel
    vW = 100
    v_panel = np.full((h, vW), 255, dtype=np.uint8)
    if vproj.size:
        vp = vproj / (vproj.max() + 1e-6)
        for i in range(vW):
            sx = int(round(i / max(1, vW-1) * max(0, w-1)))
            bar = int(round(vp[sx] * (h-1)))
            if bar > 0:
                v_panel[h-1-bar:h-1, i] = 0
        xl_bar = int(round(max(0, xl) / max(1, w-1) * max(0, vW-1)))
        xr_bar = int(round(max(0, xr-1) / max(1, w-1) * max(0, vW-1)))
        cv2.line(v_panel, (xl_bar, 0), (xl_bar, h-1), 128, 1)
        cv2.line(v_panel, (xr_bar, 0), (xr_bar, h-1), 128, 1)
    v_panel = cv2.cvtColor(v_panel, cv2.COLOR_GRAY2BGR)

    # horizontal panel
    hH = 100
    h_panel = np.full((hH, w), 255, dtype=np.uint8)
    if hproj.size:
        hp = hproj / (hproj.max() + 1e-6)
        for i in range(hH):
            sy = int(round(i / max(1, hH-1) * max(0, h-1)))
            bar = int(round(hp[sy] * (w-1)))
            if bar > 0:
                h_panel[i, :bar] = 0
        yt_bar = int(round(max(0, yt) / max(1, h-1) * max(0, hH-1)))
        yb_bar = int(round(max(0, yb-1) / max(1, h-1) * max(0, hH-1)))
        cv2.line(h_panel, (0, yt_bar), (w-1, yt_bar), 128, 1)
        cv2.line(h_panel, (0, yb_bar), (w-1, yb_bar), 128, 1)
    h_panel = cv2.cvtColor(h_panel, cv2.COLOR_GRAY2BGR)

    top = np.hstack([vis, v_panel])
    blank_bottom = np.full((hH, vW, 3), 255, dtype=np.uint8)
    bottom = np.hstack([h_panel, blank_bottom])
    return np.vstack([top, bottom])


def run_on_image(image_path: str, output_dir: str, expected_text: str | None = None,
                 framework: str = 'livetext', recognition_level: str = 'accurate',
                 language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    if _ocrmac is None:
        return {'success': False, 'error': f'ocrmac 未安装: {_OCRMAC_IMPORT_ERROR}'}
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': f'cannot read: {image_path}'}
    os.makedirs(output_dir, exist_ok=True)
    langs = language_preference or ['zh-Hans']
    try:
        o = _ocrmac.OCR(image_path, framework='livetext', recognition_level=recognition_level,
                        language_preference=langs, detail=True)
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
        x = int(round(nb[0] * W))
        y_top = int(round((1.0 - nb[1] - nb[3]) * H))
        w = int(round(nb[2] * W))
        h = int(round(nb[3] * H))
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
    boxes.sort(key=lambda b: (b['y'], b['x']))

    chars_meta: List[Dict[str, Any]] = []
    refined_boxes: List[Tuple[int,int,int,int]] = []
    dbg_enabled = bool(SEGMENT_REFINE_CONFIG.get('debug_visualize', False))
    dbg_dirname = str(SEGMENT_REFINE_CONFIG.get('debug_dirname', 'debug'))

    for b in boxes:
        ex = int(max(0, SEGMENT_REFINE_CONFIG.get('expand_px', 0)))
        x0 = max(0, b['x'] - ex); y0 = max(0, b['y'] - ex)
        x1 = min(W, b['x'] + b['w'] + ex); y1 = min(H, b['y'] + b['h'] + ex)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        xl, xr, yt, yb = trim_borders(gray, BORDER_TRIM_CONFIG, debug=False)
        fp = int(SEGMENT_REFINE_CONFIG.get('final_pad', 0))
        xl = max(0, xl - fp); xr = min(gray.shape[1], xr + fp)
        yt = max(0, yt - fp); yb = min(gray.shape[0], yb + fp)
        rx, ry, rw, rh = x0 + int(xl), y0 + int(yt), max(1, int(xr - xl)), max(1, int(yb - yt))
        refined_boxes.append((rx, ry, rw, rh))
        crop = img[ry:ry+rh, rx:rx+rw]
        fname = f"char_{b['order']:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), crop)
        if dbg_enabled:
            try:
                dbg_img = _render_border_debug(
                    roi,
                    int(xl), int(xr), int(yt), int(yb),
                    bin_mode=str(BORDER_TRIM_CONFIG.get('binarize', 'otsu')).lower(),
                    ablock=int(BORDER_TRIM_CONFIG.get('adaptive_block', 31)),
                    aC=int(BORDER_TRIM_CONFIG.get('adaptive_C', 3)),
                )
                out_dbg = os.path.join(output_dir, dbg_dirname)
                _ensure_dir(out_dbg)
                cv2.imwrite(os.path.join(out_dbg, f"{os.path.splitext(fname)[0]}_proj.png"), dbg_img)
            except Exception:
                pass
        chars_meta.append({
            'filename': fname,
            'bbox': (b['x'], b['y'], b['w'], b['h']),
            'refined_bbox': (rx, ry, rw, rh),
            'text_hint': b['text'],
            'confidence': b['confidence'],
            'normalized_bbox': b.get('normalized_bbox')
        })

    # overlay
    overlay = img.copy()
    palette = [(255,0,0),(0,160,255),(0,200,0),(180,0,200),(50,50,255),(255,140,0)]
    for idx, b in enumerate(boxes):
        c = palette[(b['order']-1)%len(palette)]
        cv2.rectangle(overlay, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), c, 1)
        rx, ry, rw, rh = refined_boxes[idx]
        cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), c, 2)
        cv2.putText(overlay, str(b['order']), (rx+2, ry+rh-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    overlay_path = os.path.join(output_dir, 'overlay.png')
    cv2.imwrite(overlay_path, overlay)

    stats = {
        'framework': 'livetext',
        'recognition_level': recognition_level,
        'char_candidates': len(boxes),
        'expected_text_len': len(expected_text) if expected_text else None,
        'image_size': [W, H],
        'refine_config': {
            'mode': 'border',
            'expand_px': int(SEGMENT_REFINE_CONFIG.get('expand_px', 0)),
            'final_pad': int(SEGMENT_REFINE_CONFIG.get('final_pad', 0)),
        },
        'border_trim': BORDER_TRIM_CONFIG,
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
            res = run_on_image(img_path, out_dir, expected_text=exp_text, framework=framework,
                               recognition_level=recognition_level, language_preference=language_preference)
            with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            processed.append({'dataset': ds, 'region': region_name, 'out_dir': out_dir,
                              'count': res.get('character_count', 0)})
        except Exception as e:
            errors.append({'dataset': ds, 'region': region_name, 'error': str(e)})
    return {'processed': processed, 'errors': errors}

