"""
Robust vertical single-character segmentation for OCR region images.

Designed and tuned for data/results/preocr/demo/region_images/region_001.jpg.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple, Dict, Any

import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.signal import find_peaks

"""Fail-fast config import and stable sys.path setup"""
# Ensure repository root (the folder that contains the 'src' package) is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Strict import: do not silently fallback
try:
    from src.config import SEGMENT_REFINEMENT_CONFIG, PREOCR_DIR, SEGMENTS_DIR
except Exception as e:
    raise RuntimeError(
        f"无法导入 src.config（SEGMENT_REFINEMENT_CONFIG/PREOCR_DIR/SEGMENTS_DIR）。\n"
        f"请从仓库根目录运行或将仓库根目录加入 PYTHONPATH。原始错误: {e}"
    )


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 10)
    return bin_img


def _crop_horizontal_foreground(binary: np.ndarray, margin: int = 2) -> Tuple[np.ndarray, Tuple[int, int]]:
    vproj = np.sum(binary, axis=0)
    cols = np.where(vproj > 0)[0]
    if len(cols) == 0:
        return binary, (0, binary.shape[1])
    x1 = max(0, cols[0] - margin)
    x2 = min(binary.shape[1], cols[-1] + margin + 1)
    return binary[:, x1:x2], (x1, x2)


def _deskew_small_angles(gray: np.ndarray, binary: np.ndarray, angle_range=(-3.0, 3.0), step=0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = gray.shape
    best_score = -1
    best_angle = 0.0
    best_gray = gray
    best_bin = binary

    def score_of(bin_img: np.ndarray) -> float:
        proj = np.sum(bin_img, axis=1) // 255
        sm = gaussian_filter1d(proj.astype(float), sigma=max(1.0, h / 200))
        inv = np.max(sm) - sm
        peaks, props = find_peaks(inv, distance=max(3, int(h / 50)), prominence=np.max(inv) * 0.05)
        if len(peaks) == 0:
            return 0.0
        prom = np.mean(props.get('prominences', np.array([0.0])))
        return len(peaks) * prom

    for ang in np.arange(angle_range[0], angle_range[1] + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        g_r = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b_r = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sc = score_of(b_r)
        if sc > best_score:
            best_score, best_angle, best_gray, best_bin = sc, ang, g_r, b_r
    return best_gray, best_bin, best_angle


def _horizontal_gaps(bin_img: np.ndarray, mean_char_h_hint: float | None = None) -> List[int]:
    proj = np.sum(bin_img, axis=1) // 255
    h = len(proj)
    sigma = max(1.5, (mean_char_h_hint or (h / 20)) / 6)
    sm = gaussian_filter1d(proj.astype(float), sigma=sigma)
    inv = np.max(sm) - sm
    min_dist = max(3, int((mean_char_h_hint or (h / 20)) / 2))
    peaks, _ = find_peaks(inv, distance=min_dist, prominence=np.max(inv) * 0.05)
    return peaks.tolist()


def _segments_from_gaps(h: int, gaps: List[int], min_seg_h: int) -> List[Tuple[int, int]]:
    if not gaps:
        return [(0, h)]
    segs: List[Tuple[int, int]] = []
    prev = 0
    for g in gaps:
        if g - prev >= min_seg_h:
            segs.append((prev, g))
            prev = g
    if h - prev >= min_seg_h:
        segs.append((prev, h))
    return segs


def _watershed_split(bin_img: np.ndarray) -> List[Tuple[int, int]]:
    h, w = bin_img.shape
    dist = cv2.distanceTransform(bin_img, distanceType=cv2.DIST_L2, maskSize=5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(bin_img, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    segs: List[Tuple[int, int]] = []
    for m in range(2, num_markers + 1):
        ys, xs = np.where(markers == m)
        if len(ys) == 0:
            continue
        y1, y2 = int(np.min(ys)), int(np.max(ys) + 1)
        if y2 - y1 >= 4:
            segs.append((y1, y2))
    segs = sorted(segs, key=lambda t: t[0])
    merged: List[Tuple[int, int]] = []
    for s in segs:
        if not merged or s[0] > merged[-1][1] + 1:
            merged.append(s)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
    return merged


def _compute_cost_maps(gray: np.ndarray, bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel costs for dynamic seam.
    Returns: ink_cost (H,W), dist_cost (H,W), grad_cost (H,W)
    """
    # Ink cost: prefer passing through background (bin==0), penalize foreground
    ink_cost = (bin_img.astype(np.float32) / 255.0)
    # Distance transform on background (distance to ink); larger distance => cheaper
    bg = (bin_img == 0).astype(np.uint8)
    dist = distance_transform_edt(bg).astype(np.float32)
    if dist.max() > 1e-6:
        dist_norm = 1.0 - (dist / (dist.max() + 1e-6))  # want smaller cost where distance is larger
    else:
        dist_norm = np.ones_like(dist, dtype=np.float32)
    # Gradient magnitude from gray image
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    if grad.max() > 1e-6:
        grad_norm = grad / (grad.max() + 1e-6)
    else:
        grad_norm = np.zeros_like(grad, dtype=np.float32)
    return ink_cost, dist_norm, grad_norm


def _dynamic_seam(bin_img: np.ndarray, gray: np.ndarray, y_split: int, band: int) -> List[Tuple[int, int]]:
    """Compute a vertical seam (x varying with y) around horizontal y_split within ±band window.
    Returns list of (x,y) coordinates for the seam path.
    """
    h, w = bin_img.shape
    y0 = max(0, y_split - band)
    y1 = min(h - 1, y_split + band)
    if y1 <= y0:
        mid_x = w // 2
        return [(mid_x, y) for y in range(y0, y1 + 1)]

    ink_c, dist_c, grad_c = _compute_cost_maps(gray, bin_img)
    wi = SEGMENT_REFINEMENT_CONFIG['seam_ink_weight']
    wd = SEGMENT_REFINEMENT_CONFIG['seam_dist_weight']
    wg = SEGMENT_REFINEMENT_CONFIG['seam_grad_weight']
    cost = wi * ink_c + wd * dist_c + wg * grad_c

    # DP: find minimal-cost path from top row y0 to bottom row y1
    band_h = y1 - y0 + 1
    dp = np.full((band_h, w), np.float32(1e9))
    prv = np.full((band_h, w), -1, dtype=np.int16)
    # initialize middle row with costs; bias towards column with least cost
    dp[0, :] = cost[y0, :]
    for i in range(1, band_h):
        yy = y0 + i
        for x in range(w):
            # allow small horizontal moves: -1,0,+1
            best_val = dp[i - 1, x]
            best_k = x
            if x > 0 and dp[i - 1, x - 1] < best_val:
                best_val = dp[i - 1, x - 1]
                best_k = x - 1
            if x + 1 < w and dp[i - 1, x + 1] < best_val:
                best_val = dp[i - 1, x + 1]
                best_k = x + 1
            dp[i, x] = best_val + cost[yy, x]
            prv[i, x] = best_k

    # Backtrack from minimal cost at bottom row
    end_x = int(np.argmin(dp[-1, :]))
    path = []
    x = end_x
    for i in range(band_h - 1, -1, -1):
        y = y0 + i
        path.append((x, y))
        x = int(prv[i, x]) if i > 0 else x
        if x < 0:
            x = 0
    path.reverse()
    return path


def _cut_with_seams(image: np.ndarray, gray: np.ndarray, bin_img: np.ndarray,
                    segs: List[Tuple[int, int]], mean_hint: int) -> List[Tuple[int, int, int, int]]:
    """Refine straight horizontal cuts into minimal-cost seams and return boxes.
    """
    h, w = bin_img.shape
    band = max(1, int(mean_hint * SEGMENT_REFINEMENT_CONFIG['seam_band_ratio']))
    boxes: List[Tuple[int, int, int, int]] = []
    for (y1, y2) in sorted(segs, key=lambda t: t[0]):
        # draw seam at the top boundary y1
        y_split = y1
        seam = _dynamic_seam(bin_img, gray, y_split, band)
        # find min/max x along seam to ensure a safe left bound; here we keep a full column box
        # For simplicity in this version, output straight boxes as before (x1..x2),
        # but seam can be visualized in debug overlay.
        boxes.append((0, max(0, y1), w, max(1, min(h, y2) - max(0, y1))))
    return boxes


def _compute_split_seams(gray: np.ndarray, bin_img: np.ndarray,
                         segs: List[Tuple[int, int]], mean_hint: int) -> List[List[Tuple[int, int]]]:
    """Compute seams on the split boundaries between adjacent segments.
    Returns a list of polylines (list of (x,y) in cropped coordinates).
    """
    if not segs or len(segs) <= 1:
        return []
    band = max(1, int(mean_hint * SEGMENT_REFINEMENT_CONFIG['seam_band_ratio']))
    boundaries = [segs[i][0] for i in range(1, len(segs))]
    seams: List[List[Tuple[int, int]]] = []
    for y_split in boundaries:
        seams.append(_dynamic_seam(bin_img, gray, y_split, band))
    return seams


def segment_vertical_single_column(image: np.ndarray,
                                   expected_chars: int | None = None,
                                   debug: bool = True,
                                   enable_dynamic_seam: bool | None = None,
                                   align_expected_count: bool | None = None) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    h, w = image.shape[:2]
    gray = _to_gray(image)
    gray = _clahe(gray)
    bin0 = _adaptive_binarize(gray)
    bin1, (x1, x2) = _crop_horizontal_foreground(bin0)
    gray_cropped = gray[:, x1:x2]

    gray2, bin2, angle = _deskew_small_angles(gray_cropped, bin1)

    mean_hint = max(12, int(h / 20))
    gaps = _horizontal_gaps(bin2, mean_char_h_hint=mean_hint)
    min_seg_h = max(6, int((h / 30)))
    segs = _segments_from_gaps(bin2.shape[0], gaps, min_seg_h=min_seg_h)

    if expected_chars is not None and expected_chars > 0 and len(segs) < max(2, expected_chars // 2):
        ws = _watershed_split(bin2)
        if len(ws) > len(segs):
            segs = ws

    # Expected count alignment (simple heuristic split/merge)
    if ((align_expected_count if align_expected_count is not None else SEGMENT_REFINEMENT_CONFIG['expected_count_alignment'])
            and (expected_chars is not None) and (expected_chars > 0)):
        # If segments are fewer than expected, try splitting the tallest ones at their weakest valley
        split_attempts = 0
        while len(segs) < expected_chars and split_attempts < SEGMENT_REFINEMENT_CONFIG['max_split_attempts']:
            # pick segment with largest height
            idx = int(np.argmax([b - a for (a, b) in segs])) if segs else -1
            if idx < 0:
                break
            a, b = segs[idx]
            band_h = b - a
            if band_h < max(6, int(mean_hint * 0.8)):
                break
            # find a local minimum within [a,b]
            sub = np.sum(bin2[a:b, :] // 255, axis=1).astype(float)
            sm = gaussian_filter1d(sub, sigma=max(1.0, band_h / 15))
            inv = np.max(sm) - sm
            valley = int(np.argmin(inv)) + a
            if valley <= a + 2 or valley >= b - 2:
                break
            new_segs = segs[:idx] + [(a, valley), (valley, b)] + segs[idx + 1:]
            segs = sorted(new_segs, key=lambda t: t[0])
            split_attempts += 1

        # If segments are more than expected, try merging the shortest gaps
        merge_attempts = 0
        while len(segs) > expected_chars and merge_attempts < SEGMENT_REFINEMENT_CONFIG['max_merge_attempts']:
            # merge pair that yields minimal height increase (closest neighbors)
            best_i = -1
            best_gap = 1e9
            for i in range(len(segs) - 1):
                gap = segs[i + 1][0] - segs[i][1]
                if gap < best_gap:
                    best_gap = gap
                    best_i = i
            if best_i < 0:
                break
            a1, b1 = segs[best_i]
            a2, b2 = segs[best_i + 1]
            merged = (a1, b2)
            segs = segs[:best_i] + [merged] + segs[best_i + 2:]
            merge_attempts += 1

    # Convert to boxes within cropped x-range
    use_seam = (enable_dynamic_seam if enable_dynamic_seam is not None else SEGMENT_REFINEMENT_CONFIG['enable_dynamic_seam'])
    if use_seam:
        # Use dynamic seam only for overlay (boxes remain rectangles for now)
        boxes_inner = _cut_with_seams(image[:, x1:x2], gray2, bin2, segs, mean_hint)
    else:
        boxes_inner = [(0, max(0, y1), bin2.shape[1], max(1, min(bin2.shape[0], y2) - max(0, y1))) for (y1, y2) in segs]

    boxes: List[Tuple[int, int, int, int]] = []
    for (xx, y, ww, hh) in boxes_inner:
        y1c = max(0, min(y, h - 1))
        y2c = max(y1c + 1, min(y + hh, h))
        boxes.append((x1 + xx, y1c, min(x2 - x1, ww), y2c - y1c))

    stats = {
        'angle': float(angle),
        'segments': int(len(boxes)),
        'x_crop': (int(x1), int(x2)),
    }

    # Prepare overlay seams (global coordinates) for drawing
    if use_seam and SEGMENT_REFINEMENT_CONFIG.get('debug_overlay', True):
        seams_cropped = _compute_split_seams(gray2, bin2, segs, mean_hint)
        seams_global: List[List[Tuple[int, int]]] = []
        for poly in seams_cropped:
            # downsample points to reduce size
            ds_poly = poly[::2] if len(poly) > 1 else poly
            seams_global.append([(x1 + int(px), int(py)) for (px, py) in ds_poly])
        stats['_overlay_seams'] = seams_global

    return boxes, stats


def run_on_image(image_path: str, output_dir: str, expected_text: str | None = None,
                 enable_dynamic_seam: bool | None = None,
                 align_expected_count: bool | None = None,
                 crop_mode_override: str | None = None) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': f'cannot read: {image_path}'}
    expected = len(expected_text) if expected_text else None
    boxes, stats = segment_vertical_single_column(
        img,
        expected_chars=expected,
        debug=True,
        enable_dynamic_seam=enable_dynamic_seam,
        align_expected_count=align_expected_count,
    )

    # optional tight-crop for each char chip
    try:
        from src.config import CHAR_CROP_CONFIG, CHAR_NOISE_CLEAN_CONFIG
    except Exception as e:
        raise RuntimeError(
            f"无法从 src.config 导入 CHAR_CROP_CONFIG/CHAR_NOISE_CLEAN_CONFIG。\n"
            f"请确认 src 包与配置文件可被导入。原始错误: {e}"
        )

    effective_crop_mode = str(crop_mode_override) if crop_mode_override else str(CHAR_CROP_CONFIG.get('mode', 'content'))

    def _tight_crop(bgr: np.ndarray) -> np.ndarray:
        if not CHAR_CROP_CONFIG.get('enabled', True):
            return bgr
        gray = bgr[:, :, 0] if bgr.ndim == 3 and bgr.shape[2] == 1 else (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr)

        # margin-only: only trim surrounding white borders, no morphology or CC filtering
        if effective_crop_mode == 'margin_only':
            thr = 245  # treat near-white as background
            mask = gray < thr
            ys, xs = np.where(mask)
            if ys.size < int(CHAR_CROP_CONFIG.get('min_fg_area', 8)):
                return bgr
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            pad = int(CHAR_CROP_CONFIG.get('pad', 2))
            y1 = max(0, y1 - pad); y2 = min(bgr.shape[0], y2 + pad)
            x1 = max(0, x1 - pad); x2 = min(bgr.shape[1], x2 + pad)
            cropped = bgr[y1:y2, x1:x2]
            fp = int(CHAR_CROP_CONFIG.get('final_padding', 0))
            sq = bool(CHAR_CROP_CONFIG.get('square_output', False))
            if fp > 0 or sq:
                h2, w2 = cropped.shape[:2]
                if sq:
                    side = max(h2, w2) + 2 * fp
                    out = np.full((side, side, 3), 255, dtype=cropped.dtype)
                    y_off = (side - h2) // 2
                    x_off = (side - w2) // 2
                    out[y_off:y_off + h2, x_off:x_off + w2] = cropped
                    return out
                else:
                    out = np.full((h2 + 2 * fp, w2 + 2 * fp, 3), 255, dtype=cropped.dtype)
                    out[fp:fp + h2, fp:fp + w2] = cropped
                    return out
            return cropped

        # content-based crop with optional noise cleaning
        if CHAR_CROP_CONFIG.get('binarize', 'otsu') == 'adaptive':
            bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        else:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # optional morphology
        k = np.ones((3, 3), np.uint8)
        op = int(CHAR_NOISE_CLEAN_CONFIG.get('morph_open', 0))
        cl = int(CHAR_NOISE_CLEAN_CONFIG.get('morph_close', 0))
        if op > 0:
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=op)
        if cl > 0:
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=cl)

        # connected components to remove small specks
        if CHAR_NOISE_CLEAN_CONFIG.get('enabled', True):
            num, labels, stats, centroids = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8), connectivity=8)
            if num > 1:
                areas = stats[:, cv2.CC_STAT_AREA]
                idxs = list(range(1, num))
                h0, w0 = bin_img.shape[:2]
                min_area_abs = int(CHAR_NOISE_CLEAN_CONFIG.get('min_area', 10))
                min_area_rel = int(CHAR_NOISE_CLEAN_CONFIG.get('min_area_ratio', 0.002) * h0 * w0)
                min_area = max(1, max(min_area_abs, min_area_rel))
                kept = []
                cx, cy = w0 / 2.0, h0 / 2.0
                center_bias = float(CHAR_NOISE_CLEAN_CONFIG.get('center_bias', 0.0))
                best_label = None
                best_score = -1e18
                for i in idxs:
                    a = int(areas[i])
                    if a < min_area:
                        continue
                    (mx, my) = centroids[i]
                    score = a - center_bias * ((mx - cx) ** 2 + (my - cy) ** 2)
                    if score > best_score:
                        best_score = score
                        best_label = i
                if best_label is not None:
                    kept.append(best_label)
                    sec_ratio = float(CHAR_NOISE_CLEAN_CONFIG.get('second_keep_ratio', 0.35))
                    sorted_idxs = sorted([i for i in idxs if int(areas[i]) >= min_area], key=lambda i: int(areas[i]), reverse=True)
                    if len(sorted_idxs) >= 2:
                        largest = sorted_idxs[0]
                        second = sorted_idxs[1]
                        if second != best_label and areas[second] >= areas[largest] * sec_ratio:
                            kept.append(second)
                clean = np.zeros_like(bin_img)
                for lb in kept:
                    clean[labels == lb] = 255
                if clean.any():
                    bin_img = clean
        ys, xs = np.where(bin_img > 0)
        if ys.size < CHAR_CROP_CONFIG.get('min_fg_area', 8):
            return bgr
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        pad = int(CHAR_CROP_CONFIG.get('pad', 2))
        y1 = max(0, y1 - pad); y2 = min(bgr.shape[0], y2 + pad)
        x1 = max(0, x1 - pad); x2 = min(bgr.shape[1], x2 + pad)
        cropped = bgr[y1:y2, x1:x2]
        try:
            sub_mask = bin_img[y1:y2, x1:x2]
            ys2, xs2 = np.where(sub_mask > 0)
            if ys2.size >= CHAR_CROP_CONFIG.get('min_fg_area', 8):
                y1b, y2b = int(ys2.min()), int(ys2.max()) + 1
                x1b, x2b = int(xs2.min()), int(xs2.max()) + 1
                pad2 = int(CHAR_CROP_CONFIG.get('pad', 2))
                y1b = max(0, y1b - pad2); y2b = min(cropped.shape[0], y2b + pad2)
                x1b = max(0, x1b - pad2); x2b = min(cropped.shape[1], x2b + pad2)
                cropped = cropped[y1b:y2b, x1b:x2b]
        except Exception:
            pass
        fp = int(CHAR_CROP_CONFIG.get('final_padding', 0))
        sq = bool(CHAR_CROP_CONFIG.get('square_output', False))
        if fp > 0 or sq:
            h3, w3 = cropped.shape[:2]
            if sq:
                side = max(h3, w3) + 2 * fp
                out = np.zeros((side, side, 3), dtype=cropped.dtype)
                bg = 255
                out[:] = (bg, bg, bg)
                y_off = (side - h3) // 2
                x_off = (side - w3) // 2
                out[y_off:y_off + h3, x_off:x_off + w3] = cropped
                return out
            else:
                out = np.zeros((h3 + 2 * fp, w3 + 2 * fp, 3), dtype=cropped.dtype)
                bg = 255
                out[:] = (bg, bg, bg)
                out[fp:fp + h3, fp:fp + w3] = cropped
                return out
        return cropped

    os.makedirs(output_dir, exist_ok=True)
    chars = []
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = img[y:y + h, x:x + w]
        char_img = _tight_crop(char_img)
        name = f"char_{i + 1:04d}.png"
        cv2.imwrite(os.path.join(output_dir, name), char_img)
        chars.append({'filename': name, 'bbox': (int(x), int(y), int(w), int(h))})

    # debug overlay
    overlay = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # draw seams if available
    seams = stats.pop('_overlay_seams', None)
    if seams:
        for poly in seams:
            if len(poly) >= 2:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
    cv2.imwrite(os.path.join(output_dir, 'overlay.png'), overlay)

    # enrich stats with effective crop/noise settings for debugging
    try:
        stats['crop_mode'] = effective_crop_mode
        stats['crop_enabled'] = bool(CHAR_CROP_CONFIG.get('enabled', True))
        stats['crop_binarize'] = str(CHAR_CROP_CONFIG.get('binarize', 'otsu'))
        stats['crop_pad'] = int(CHAR_CROP_CONFIG.get('pad', 2))
        eff_noise = False if (effective_crop_mode == 'margin_only' or not CHAR_CROP_CONFIG.get('enabled', True)) else bool(CHAR_NOISE_CLEAN_CONFIG.get('enabled', True))
        stats['noise_clean_enabled'] = eff_noise
        stats['noise_morph_open'] = int(CHAR_NOISE_CLEAN_CONFIG.get('morph_open', 0))
        stats['noise_morph_close'] = int(CHAR_NOISE_CLEAN_CONFIG.get('morph_close', 0))
    except Exception:
        pass

    return {
        'success': True,
        'character_count': len(chars),
        'characters': chars,
        'stats': stats,
        'overlay': os.path.join(output_dir, 'overlay.png')
    }


def _find_region_images(base_ocr_dir: str, dataset: str | None = None) -> List[Tuple[str, str, str]]:
    """Scan OCR_DIR for region images.
    Returns list of (dataset, region_name, image_path)
    """
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
    # deterministic order
    results.sort(key=lambda t: (t[0], t[1]))
    return results


def run_on_ocr_regions(dataset: str | None = None,
                       expected_texts: Dict[str, str] | None = None,
                       enable_dynamic_seam: bool | None = None,
                       align_expected_count: bool | None = None,
                       crop_mode_override: str | None = None) -> Dict[str, Any]:
    def _to_py(x):
        import numpy as _np
        if isinstance(x, dict):
            return {k: _to_py(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [ _to_py(v) for v in x ]
        if isinstance(x, (_np.generic,)):
            return x.item()
        return x
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
            res = run_on_image(img_path, out_dir, expected_text=exp_text,
                               enable_dynamic_seam=enable_dynamic_seam,
                               align_expected_count=align_expected_count,
                               crop_mode_override=crop_mode_override)
            # write per-region summary
            with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(_to_py(res), f, ensure_ascii=False, indent=2)
            processed.append({'dataset': ds, 'region': region_name, 'out_dir': out_dir, 'count': res.get('character_count', 0)})
        except Exception as e:
            errors.append({'dataset': ds, 'region': region_name, 'error': str(e)})
    return {'processed': processed, 'errors': errors}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Vertical hybrid segmentation with optional dynamic seam and expected-count alignment')
    parser.add_argument('--image', default='data/results/preocr/demo/region_images/region_001.jpg')
    parser.add_argument('--out', default='data/results/segments/demo/region_001')
    parser.add_argument('--expected-text', default=None)
    parser.add_argument('--no-seam', action='store_true', help='Disable dynamic seam refinement')
    parser.add_argument('--no-align', action='store_true', help='Disable expected count alignment')
    parser.add_argument('--scan-ocr', action='store_true', help='Scan OCR outputs and process all region images')
    parser.add_argument('--dataset', default=None, help='Restrict to a specific dataset name when using --scan-ocr')
    parser.add_argument('--crop-mode', default=None, choices=['content', 'margin_only'], help='Override char crop mode for this run')
    args = parser.parse_args()

    if args.scan_ocr:
        summary = run_on_ocr_regions(
            dataset=args.dataset,
            enable_dynamic_seam=(False if args.no_seam else None),
            align_expected_count=(False if args.no_align else None),
            crop_mode_override=args.crop_mode,
        )
        print(json.dumps({'processed': len(summary['processed']), 'errors': summary['errors']}, ensure_ascii=False, indent=2))
    else:
        os.makedirs(args.out, exist_ok=True)
        res = run_on_image(
            args.image,
            args.out,
            expected_text=args.expected_text,
            enable_dynamic_seam=(False if args.no_seam else None),
            align_expected_count=(False if args.no_align else None),
            crop_mode_override=args.crop_mode,
        )
        # sanitize numpy types for safe JSON serialization
        def _to_py(obj):
            import numpy as _np
            if isinstance(obj, dict):
                return {k: _to_py(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [ _to_py(v) for v in obj ]
            if isinstance(obj, (_np.generic,)):
                return obj.item()
            return obj
        print(json.dumps(_to_py(res), ensure_ascii=False, indent=2))
