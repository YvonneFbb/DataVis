"""
Legacy segmentation helpers migrated out of the main pipeline.

This file preserves older projection/grabcut/morph refinement utilities for
reference. They are not used by the current pipeline which relies on
border-first trimming in `vertical_hybrid.py` + `border_trim.py`.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import cv2


def cut_by_valley(proj: np.ndarray, trim_left_max: int, trim_right_max: int,
                  valley_ratio: float, valley_abs: float) -> Tuple[int, int]:
    n = int(proj.shape[0])
    if n <= 0:
        return 0, 0
    thr = max(float(valley_abs), float(np.median(proj)) * float(valley_ratio))
    mask = proj > thr
    best_l = 0; best_r = n; best_mass = -1.0
    i = 0
    while i < n:
        if mask[i]:
            j = i; mass = 0.0
            while j < n and mask[j]:
                mass += float(proj[j]); j += 1
            if mass > best_mass:
                best_mass = mass; best_l = i; best_r = j
            i = j
        else:
            i += 1
    if best_mass < 0:
        return 0, n
    trim_left = min(best_l, int(trim_left_max))
    trim_right = min(max(0, n - best_r), int(trim_right_max))
    L = max(0, trim_left)
    R = min(n, n - trim_right)
    if R <= L:
        return 0, n
    return L, R


def refine_edges_by_drop(proj: np.ndarray, L: int, R: int,
                         hi_ratio: float, lo_ratio: float,
                         min_grad_ratio: float,
                         min_high_run: int,
                         search_window: int,
                         trim_left_max: int, trim_right_max: int) -> Tuple[int, int]:
    n = int(proj.shape[0])
    if n <= 1:
        return L, R
    med = float(np.median(proj))
    pmax = float(np.max(proj)) if np.max(proj) > 0 else 1.0
    thr_hi = med * float(hi_ratio)
    thr_lo = med * float(lo_ratio)
    pad = np.pad(proj.astype(np.float32), (1,1), mode='edge')
    grad = (pad[2:] - pad[:-2]) * 0.5
    if n >= 5:
        k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        grad = np.convolve(grad, k, mode='same')
    min_grad = float(min_grad_ratio) * pmax
    W = int(max(1, search_window))
    # left
    l0 = max(0, int(L) - W); l1 = min(n-2, int(L) + W)
    bestL = L; bestGL = 0.0
    for i in range(l0, l1+1):
        g = float(grad[i])
        if g < min_grad:
            continue
        pre_ok = float(np.mean(proj[max(0, i-3):i+1])) <= thr_lo
        post_seg = proj[i+1:min(n, i+1+min_high_run)]
        post_ok = post_seg.size >= 1 and float(np.min(post_seg)) >= thr_hi
        if pre_ok and post_ok and g > bestGL:
            bestGL = g; bestL = i+1
    bestL = int(min(max(0, bestL), min(n, L + trim_left_max)))
    # right
    r0 = max(0, int(R) - W); r1 = min(n-2, int(R) + W)
    bestR = R; bestGR = 0.0
    for i in range(r0, r1+1):
        g = float(-grad[i])
        if g < min_grad:
            continue
        pre_seg = proj[max(0, i-min_high_run+1):i+1]
        pre_ok = pre_seg.size >= 1 and float(np.min(pre_seg)) >= thr_hi
        post_ok = float(np.mean(proj[i+1:min(n, i+4)])) <= thr_lo
        if pre_ok and post_ok and g > bestGR:
            bestGR = g; bestR = i+1
    bestR = int(max(0, min(n, max(bestR, R - trim_right_max))))
    if bestR <= bestL:
        return L, R
    return bestL, bestR


def legacy_binarize(gray: np.ndarray, mode: str = 'adaptive', block: int = 21, C: int = 5) -> np.ndarray:
    if mode == 'adaptive':
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, block, C)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img


def legacy_refine_bbox_projection(img, x, y, w, h, config: Dict[str, Any]) -> Tuple[int,int,int,int]:
    H, W = img.shape[:2]
    ex = int(max(0, config.get('expand_px', 0)))
    x0 = max(0, x - ex); y0 = max(0, y - ex)
    x1 = min(W, x + w + ex); y1 = min(H, y + h + ex)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return x, y, w, h
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bin_img = legacy_binarize(gray, mode=config.get('binarize', 'adaptive'),
                              block=int(config.get('adaptive_block', 21)),
                              C=int(config.get('adaptive_C', 5)))
    # strip near-edge thin lines (simplified)
    if bool(config.get('strip_border_lines', True)):
        search_px = int(max(1, config.get('strip_search_px', 6)))
        max_w = int(max(1, config.get('strip_max_width', 3)))
        min_cov = float(config.get('strip_min_coverage', 0.6))
        h_roi, w_roi = bin_img.shape
        left = bin_img[:, :min(search_px, w_roi)]
        right = bin_img[:, max(0, w_roi - search_px):]
        def _strip_band(band, offset_x):
            num, labels, stats, _ = cv2.connectedComponentsWithStats((band>0).astype(np.uint8), connectivity=8)
            removed = np.zeros_like(band)
            for i in range(1, num):
                x_i, y_i, w_i, h_i, _area_i = stats[i]
                if w_i <= max_w and (h_i / band.shape[0]) >= min_cov:
                    removed[labels == i] = 255
            nz = np.where(removed>0)
            if nz[0].size:
                bin_img[nz[0], nz[1] + offset_x] = 0
        _strip_band(left, 0)
        _strip_band(right, max(0, w_roi - search_px))
    # projections
    vproj = (bin_img>0).sum(axis=0).astype(np.float32)
    sm = int(config.get('proj_smooth', 1))
    if sm > 1:
        k = np.ones(sm, dtype=np.float32) / sm
        vproj = np.convolve(vproj, k, mode='same')
    trim_max_l = int(max(0, config.get('proj_trim_left_max', 12)))
    trim_max_r = int(max(0, config.get('proj_trim_right_max', 12)))
    trim_max_t = int(max(0, config.get('proj_trim_top_max', 8)))
    trim_max_b = int(max(0, config.get('proj_trim_bottom_max', 8)))
    valley_ratio = float(config.get('proj_valley_ratio', 0.05))
    valley_abs = float(config.get('proj_valley_abs', 0))
    xl, xr = cut_by_valley(vproj, trim_max_l, trim_max_r, valley_ratio, valley_abs)
    xl, xr = refine_edges_by_drop(
        vproj, xl, xr,
        float(config.get('edge_drop_hi_ratio', 0.5)),
        float(config.get('edge_drop_lo_ratio', 0.2)),
        float(config.get('edge_drop_min_grad_ratio', 0.10)),
        int(config.get('edge_drop_min_high_run', 3)),
        int(config.get('edge_drop_window', 10)),
        trim_max_l, trim_max_r,
    )
    v_slice = bin_img[:, xl:xr]
    hproj = (v_slice>0).sum(axis=1).astype(np.float32)
    if sm > 1:
        k = np.ones(sm, dtype=np.float32) / sm
        hproj = np.convolve(hproj, k, mode='same')
    yt, yb = cut_by_valley(hproj, trim_max_t, trim_max_b, valley_ratio, valley_abs)
    yt, yb = refine_edges_by_drop(
        hproj, yt, yb,
        float(config.get('edge_drop_hi_ratio', 0.5)),
        float(config.get('edge_drop_lo_ratio', 0.2)),
        float(config.get('edge_drop_min_grad_ratio', 0.10)),
        int(config.get('edge_drop_min_high_run', 3)),
        int(config.get('edge_drop_window', 10)),
        trim_max_t, trim_max_b,
    )
    w_roi = bin_img.shape[1]; h_roi = bin_img.shape[0]
    if xr <= xl: xl, xr = 0, w_roi
    if yb <= yt: yt, yb = 0, h_roi
    fp = int(config.get('final_pad', 0))
    xl = max(0, xl - fp); xr = min(w_roi, xr + fp)
    yt = max(0, yt - fp); yb = min(h_roi, yb + fp)
    return x0 + int(xl), y0 + int(yt), max(1, int(xr - xl)), max(1, int(yb - yt))


def legacy_refine_bbox_grabcut(img, x, y, w, h, config: Dict[str, Any]) -> Tuple[int,int,int,int]:
    H, W = img.shape[:2]
    ex = int(max(0, config.get('expand_px', 0)))
    x0 = max(0, x - ex); y0 = max(0, y - ex)
    x1 = min(W, x + w + ex); y1 = min(H, y + h + ex)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return x, y, w, h
    gc_iter = int(config.get('gc_iter', 3))
    inner = int(max(0, config.get('gc_inner_shrink', 2)))
    mask = np.zeros(roi.shape[:2], np.uint8)
    ih, iw = roi.shape[:2]
    mask[:,:] = cv2.GC_PR_FGD
    if iw > 2*inner and ih > 2*inner:
        mask[inner:ih-inner, inner:iw-inner] = cv2.GC_FGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(roi, mask, None, bgdModel, fgdModel, gc_iter, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return x0, y0, x1-x0, y1-y0
    m = (mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD)
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return x0, y0, x1-x0, y1-y0
    yy1, yy2 = int(ys.min()), int(ys.max())+1
    xx1, xx2 = int(xs.min()), int(xs.max())+1
    fp = int(config.get('final_pad', 0))
    yy1 = max(0, yy1 - fp); yy2 = min(roi.shape[0], yy2 + fp)
    xx1 = max(0, xx1 - fp); xx2 = min(roi.shape[1], xx2 + fp)
    return x0 + xx1, y0 + yy1, max(1, xx2-xx1), max(1, yy2-yy1)

