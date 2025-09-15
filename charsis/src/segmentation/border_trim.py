"""
Border trimming utilities: focus on cutting page/region borders using
two signals only:
  - discrete edge regions (near-edge columns with high coverage / thin lines)
  - high–low drops in 1D projections (valley + gradient evidence)

Noise cleanup, full character refinement, and other heuristics are out of scope.
This module aims to be simple, deterministic, and readable.
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import cv2


def binarize(gray: np.ndarray, mode: str = 'otsu', adaptive_block: int = 31, adaptive_C: int = 3) -> np.ndarray:
    if mode == 'adaptive':
        if adaptive_block % 2 == 0:
            adaptive_block += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    # default: otsu
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img


def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, k))
    if k <= 1:
        return x.astype(np.float32)
    ker = np.ones(k, np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode='same')


def edge_band_line_scores(bin_img: np.ndarray, side: str, band_px: int,
                          line_cov_thr: float, line_max_width: int,
                          line_min_coverage: float) -> Tuple[float, float]:
    """Compute two simple near-edge evidences:
    - edge_bias: mean column coverage in the edge band (0..1)
    - line_ratio: fraction of columns regarded as a line (coverage>=line_cov_thr)
    Also screens out connected components in the band that look like thin vertical lines.
    """
    h, w = bin_img.shape
    if w == 0 or h == 0:
        return 0.0, 0.0
    band_px = int(max(1, min(band_px, w)))
    if side == 'left':
        band = bin_img[:, :band_px]
    else:
        band = bin_img[:, max(0, w - band_px):]
    col_cov = band.sum(axis=0).astype(np.float32) / float(h * 255)
    edge_bias = float(np.mean(col_cov))
    line_ratio = float(np.mean((col_cov >= float(line_cov_thr)).astype(np.float32)))
    # Remove thin, tall connected components in the band to reduce their impact in later stages
    num, labels, stats, _ = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), connectivity=8)
    for i in range(1, num):
        x, y, ww, hh, _area = stats[i]
        cov = float(hh) / float(h)
        if ww <= int(line_max_width) and cov >= float(line_min_coverage):
            # wipe this component from the band (so downstream projections will be less biased)
            band[labels == i] = 0
    # write back wiped band to original bin_img view
    if side == 'left':
        bin_img[:, :band_px] = band
    else:
        bin_img[:, max(0, w - band_px):] = band
    return edge_bias, line_ratio


def find_cut_1d(proj: np.ndarray, side: str, params: Dict[str, float]) -> int:
    """Find a cut index on 1D projection combining valley-first + small-window
    gradient tiebreak, constrained by outside-low/inside-high evidence.
    Returns an index in [0, n] representing the cut just after that column/row.
    """
    n = int(proj.shape[0])
    if n <= 1:
        return 0 if side == 'left' else n
    p = proj.astype(np.float32)
    p_s = smooth_1d(p, int(params.get('smooth', 5)))
    # robust scales
    q25 = float(np.quantile(p_s, 0.25))
    q75 = float(np.quantile(p_s, 0.75))
    med = float(np.median(p_s))
    thr_hi = max(q75 * 0.6, med * float(params.get('hi_ratio', 0.5)))
    thr_lo = min(q25 * 1.2, med * float(params.get('lo_ratio', 0.2)))
    # search window centered at the valley produced by a coarse min search in edge half
    half = max(2, int(params.get('search_half', max(3, n // 6))))
    if side == 'left':
        coarse_zone = p_s[: max(1, int(n * 0.4))]
        valley_idx = int(np.argmin(coarse_zone))
        l0 = max(0, valley_idx - int(params.get('max_shift', 6)))
        l1 = min(n - 2, valley_idx + int(params.get('max_shift', 6)))
        cand_range = range(l0, l1 + 1)
    else:
        coarse_zone = p_s[min(n - 1, int(n * 0.6)) :]
        valley_idx = int(np.argmin(coarse_zone)) + min(n - 1, int(n * 0.6))
        r0 = max(0, valley_idx - int(params.get('max_shift', 6)))
        r1 = min(n - 2, valley_idx + int(params.get('max_shift', 6)))
        cand_range = range(r0, r1 + 1)
    # gradient
    pad = np.pad(p_s, (1, 1), mode='edge')
    grad = (pad[2:] - pad[:-2]) * 0.5
    pmax = float(np.max(p_s)) if np.max(p_s) > 1e-6 else 1.0
    min_grad = float(params.get('grad_min_ratio', 0.12)) * pmax

    def score_left(i: int) -> Tuple[float, int]:
        g = float(grad[i])
        g_s = (g / pmax) if g > 0 else 0.0
        if g < min_grad:
            g_s *= 0.2
        outside_ok = float(np.mean(p_s[max(0, i - 3) : i + 1])) <= thr_lo
        inside_run = 0
        j = i + 1
        while j < n and p_s[j] >= thr_hi:
            inside_run += 1
            j += 1
        cont = float(inside_run)
        if inside_run < int(params.get('min_high_run', 3)):
            cont *= 0.3
        valley = (pmax - float(p_s[i])) / max(1.0, pmax)
        base = 0.8 * valley + 0.5 * g_s + 0.6 * (cont / max(1.0, half))
        if not outside_ok:
            base *= 0.3
        return base, i + 1

    def score_right(i: int) -> Tuple[float, int]:
        g = -float(grad[i])
        g_s = (g / pmax) if g > 0 else 0.0
        if g < min_grad:
            g_s *= 0.2
        outside_ok = float(np.mean(p_s[i + 1 : min(n, i + 4)])) <= thr_lo
        inside_run = 0
        j = i
        while j >= 0 and p_s[j] >= thr_hi:
            inside_run += 1
            j -= 1
        cont = float(inside_run)
        if inside_run < int(params.get('min_high_run', 3)):
            cont *= 0.3
        valley = (pmax - float(p_s[i])) / max(1.0, pmax)
        base = 0.8 * valley + 0.5 * g_s + 0.6 * (cont / max(1.0, half))
        if not outside_ok:
            base *= 0.3
        return base, i + 1

    best_s = -1e9
    best_cut = 0 if side == 'left' else n
    if side == 'left':
        for i in cand_range:
            s, cut = score_left(i)
            if s > best_s:
                best_s = s
                best_cut = cut
    else:
        for i in cand_range:
            s, cut = score_right(i)
            if s > best_s:
                best_s = s
                best_cut = cut
    return int(max(0, min(n, best_cut)))


def trim_borders(gray: np.ndarray, params: Dict, debug: bool = False) -> Tuple[int, int, int, int]:
    """Return refined tight box (xl, xr, yt, yb) inside the given gray ROI
    by first wiping near-edge lines, then choosing cuts by valley-first +
    constrained gradient tiebreak.
    """
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h
    bin_img = binarize(gray, mode=str(params.get('binarize', 'otsu')).lower(),
                       adaptive_block=int(params.get('adaptive_block', 31)),
                       adaptive_C=int(params.get('adaptive_C', 3)))
    # 1) near-edge line wiping + edge evidences (not used directly in scoring here,
    #    but wiping reduces bias in projections)
    band_px = int(params.get('band_px', 10))
    line_cov_thr = float(params.get('line_cov_thr', 0.8))
    line_max_w = int(params.get('line_max_width', 4))
    line_min_cov = float(params.get('line_min_coverage', 0.8))
    # left then right
    _ = edge_band_line_scores(bin_img, 'left', band_px, line_cov_thr, line_max_w, line_min_cov)
    _ = edge_band_line_scores(bin_img, 'right', band_px, line_cov_thr, line_max_w, line_min_cov)

    # 2) projections
    vproj = (bin_img > 0).sum(axis=0).astype(np.float32)
    hproj = (bin_img > 0).sum(axis=1).astype(np.float32)

    # 3) find cuts
    pr = {
        'smooth': int(params.get('smooth', 5)),
        'hi_ratio': float(params.get('hi_ratio', 0.5)),
        'lo_ratio': float(params.get('lo_ratio', 0.2)),
        'min_high_run': int(params.get('min_high_run', 3)),
        'grad_min_ratio': float(params.get('grad_min_ratio', 0.12)),
        'max_shift': int(params.get('max_shift', 6)),
        'search_half': int(params.get('search_half', 8)),
    }
    xl = find_cut_1d(vproj, 'left', pr)
    xr = find_cut_1d(vproj, 'right', pr)
    yt = find_cut_1d(hproj, 'left', pr)
    yb = find_cut_1d(hproj, 'right', pr)

    # 4) safety
    xl = int(max(0, min(w, xl)))
    xr = int(max(xl + 1, min(w, xr)))
    yt = int(max(0, min(h, yt)))
    yb = int(max(yt + 1, min(h, yb)))
    return xl, xr, yt, yb

