"""
ocrmac-driven segmentation (CC + projection trimming, simplified).

LiveText detection → connected-component filtering → projection trimming
(src/segmentation/projection_trim.py) → save crops and debug. Legacy
projection/grabcut/morph code moved to vertical_hybrid_legacy.py and is not
used by default.
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
        SEGMENT_REFINE_CONFIG, PROJECTION_TRIM_CONFIG, CC_FILTER_CONFIG,
    )
except Exception as e:
    raise RuntimeError(
        f"无法导入必要配置 (PREOCR_DIR / SEGMENTS_DIR / SEGMENT_REFINE_CONFIG / PROJECTION_TRIM_CONFIG)。原始错误: {e}"
    )

from src.segmentation.projection_trim import trim_projection_from_bin, binarize
from src.segmentation.cc_filter import refine_binary_components


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _render_combined_debug(roi_bgr: np.ndarray,
                           bin_before: np.ndarray,
                           bin_after: np.ndarray,
                           xl: int, xr: int, yt: int, yb: int) -> np.ndarray:
    """Render 2x4 panel: CC (overlay/masks) vs Projection (overlay/projections)."""
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return roi_bgr.copy()

    mask_before = (bin_before > 0).astype(np.uint8)
    mask_after = (bin_after > 0).astype(np.uint8)

    panel_h = int(max(120, min(260, round(h * 0.8))))
    scale = panel_h / float(max(1, h))
    base_w = max(1, int(round(w * scale)))

    def resize_panel(img: np.ndarray, interp: int = cv2.INTER_LINEAR, target_w: int | None = None) -> np.ndarray:
        if target_w is None:
            target_w = base_w
        return cv2.resize(img, (target_w, panel_h), interpolation=interp)

    def to_bgr(mask: np.ndarray) -> np.ndarray:
        vis = 255 - (mask.astype(np.uint8) * 255)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def add_border(img: np.ndarray, pad: int = 2) -> np.ndarray:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # === CC overlay and masks ===
    masked_roi = np.full_like(roi_bgr, 255)
    masked_roi[mask_after.astype(bool)] = roi_bgr[mask_after.astype(bool)]
    cc_overlay = masked_roi.copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_before, connectivity=8)
    if num > 1:
        for i in range(1, num):
            x, y, ww, hh, _area = stats[i]
            kept = bool(mask_after[labels == i].any())
            color = (0, 200, 0) if kept else (0, 0, 255)
            cv2.rectangle(cc_overlay, (int(x), int(y)), (int(x + ww - 1), int(y + hh - 1)), color, 1)

    cc_overlay_panel = resize_panel(cc_overlay)
    mask_before_panel = resize_panel(to_bgr(mask_before), interp=cv2.INTER_NEAREST)
    mask_after_panel = resize_panel(to_bgr(mask_after), interp=cv2.INTER_NEAREST)

    # === Projection overlay ===
    projection_overlay = masked_roi.copy()
    cv2.line(projection_overlay, (max(0, int(xl)), 0), (max(0, int(xl)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (max(0, int(xr - 1)), 0), (max(0, int(xr - 1)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yt))), (w - 1, max(0, int(yt))), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yb - 1))), (w - 1, max(0, int(yb - 1))), (0, 0, 255), 1)
    projection_panel = resize_panel(projection_overlay)

    # === Projection histograms ===
    hist_w = base_w
    v_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    vproj = mask_after.sum(axis=0).astype(np.float32)
    if vproj.size:
        vp = vproj / (vproj.max() + 1e-6)
        for col in range(hist_w):
            sx = int(round(col / max(1, hist_w - 1) * max(0, w - 1)))
            bar = int(round(vp[sx] * (panel_h - 1)))
            if bar > 0:
                v_panel[panel_h - bar:panel_h, col] = 0
        xl_bar = int(round(max(0, xl) / max(1, w - 1) * max(0, hist_w - 1)))
        xr_bar = int(round(max(0, xr - 1) / max(1, w - 1) * max(0, hist_w - 1)))
        cv2.line(v_panel, (xl_bar, 0), (xl_bar, panel_h - 1), 128, 1)
        cv2.line(v_panel, (xr_bar, 0), (xr_bar, panel_h - 1), 128, 1)
    vertical_panel = cv2.cvtColor(v_panel, cv2.COLOR_GRAY2BGR)

    h_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    hproj = mask_after.sum(axis=1).astype(np.float32)
    if hproj.size:
        hp = hproj / (hproj.max() + 1e-6)
        for row in range(panel_h):
            sy = int(round(row / max(1, panel_h - 1) * max(0, h - 1)))
            bar = int(round(hp[sy] * (hist_w - 1)))
            if bar > 0:
                h_panel[row, :bar] = 0
        yt_bar = int(round(max(0, yt) / max(1, h - 1) * max(0, panel_h - 1)))
        yb_bar = int(round(max(0, yb - 1) / max(1, h - 1) * max(0, panel_h - 1)))
        cv2.line(h_panel, (0, yt_bar), (hist_w - 1, yt_bar), 128, 1)
        cv2.line(h_panel, (0, yb_bar), (hist_w - 1, yb_bar), 128, 1)
    horizontal_panel = cv2.cvtColor(h_panel, cv2.COLOR_GRAY2BGR)

    # === Label column ===
    label_w = max(60, hist_w // 4)
    label_color = (245, 245, 245)
    top_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    bottom_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    cv2.putText(top_label, 'CC', (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(bottom_label, 'PROJ', (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
        h_, w_ = img.shape[:2]
        if w_ >= target_w:
            return img
        pad_left = (target_w - w_) // 2
        pad_right = target_w - w_ - pad_left
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    pairs = [
        (add_border(cc_overlay_panel), add_border(projection_panel)),
        (add_border(mask_before_panel), add_border(vertical_panel)),
        (add_border(mask_after_panel), add_border(horizontal_panel)),
        (add_border(top_label), add_border(bottom_label)),
    ]

    columns = []
    for top_panel, bottom_panel in pairs:
        target_w = max(top_panel.shape[1], bottom_panel.shape[1])
        top_padded = pad_to_width(top_panel, target_w)
        bottom_padded = pad_to_width(bottom_panel, target_w)
        columns.append(np.vstack([top_padded, bottom_padded]))

    gap = 6
    column_height = columns[0].shape[0]
    gap_col = np.full((column_height, gap, 3), 255, dtype=np.uint8)

    grid = columns[0]
    for col in columns[1:]:
        grid = np.hstack([grid, gap_col, col])

    return grid

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

    raw_mode = str(SEGMENT_REFINE_CONFIG.get('mode', 'ccprojection')).lower()
    mode_alias = {
        'ccprojection': 'ccprojection',
        'cc_projection': 'ccprojection',
        'projection': 'ccprojection',
        'projection_only': 'projection_only',
        'cc_debug': 'cc_debug',
    }
    mode = mode_alias.get(raw_mode, raw_mode)
    if mode not in {'ccprojection', 'projection_only', 'cc_debug'}:
        mode = 'ccprojection'
    for b in boxes:
        ex = int(max(0, SEGMENT_REFINE_CONFIG.get('expand_px', 0)))
        x0 = max(0, b['x'] - ex); y0 = max(0, b['y'] - ex)
        x1 = min(W, b['x'] + b['w'] + ex); y1 = min(H, b['y'] + b['h'] + ex)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bin_before = binarize(
            gray,
            mode=str(PROJECTION_TRIM_CONFIG.get('binarize', 'otsu')).lower(),
            adaptive_block=int(PROJECTION_TRIM_CONFIG.get('adaptive_block', 31)),
            adaptive_C=int(PROJECTION_TRIM_CONFIG.get('adaptive_C', 3)),
        )
        # choose CC filtering according to mode
        if mode == 'projection_only':
            bin_after = bin_before.copy()  # skip CC filtering
        else:
            bin_after = bin_before.copy()
            refine_binary_components(bin_after, CC_FILTER_CONFIG)
        if mode == 'cc_debug':
            # no projection trimming; use full ROI as crop
            xl, xr, yt, yb = 0, roi.shape[1], 0, roi.shape[0]
        else:
            xl, xr, yt, yb = trim_projection_from_bin(bin_after, PROJECTION_TRIM_CONFIG)
        fp = int(SEGMENT_REFINE_CONFIG.get('final_pad', 0))
        xl = max(0, xl - fp); xr = min(roi.shape[1], xr + fp)
        yt = max(0, yt - fp); yb = min(roi.shape[0], yb + fp)
        rx, ry, rw, rh = x0 + int(xl), y0 + int(yt), max(1, int(xr - xl)), max(1, int(yb - yt))
        refined_boxes.append((rx, ry, rw, rh))
        crop = img[ry:ry+rh, rx:rx+rw]
        fname = f"char_{b['order']:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), crop)
        if dbg_enabled:
            try:
                dbg_img = _render_combined_debug(roi, bin_before, bin_after, int(xl), int(xr), int(yt), int(yb))
                dbg_name = f"{os.path.splitext(fname)[0]}_debug.png"
                out_dbg = os.path.join(output_dir, dbg_dirname)
                _ensure_dir(out_dbg)
                cv2.imwrite(os.path.join(out_dbg, dbg_name), dbg_img)
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
            'mode': mode,
            'expand_px': int(SEGMENT_REFINE_CONFIG.get('expand_px', 0)),
            'final_pad': int(SEGMENT_REFINE_CONFIG.get('final_pad', 0)),
        },
        'projection_trim': PROJECTION_TRIM_CONFIG,
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
