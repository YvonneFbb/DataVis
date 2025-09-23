"""Projection trimming based on dominant run (horizontal + vertical)."""
from __future__ import annotations
from typing import Tuple, Dict, List
import numpy as np
import cv2


def binarize(gray: np.ndarray, mode: str = 'otsu', adaptive_block: int = 31, adaptive_C: int = 3) -> np.ndarray:
    if mode == 'adaptive':
        if adaptive_block % 2 == 0:
            adaptive_block += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img


def _find_runs(coverage: np.ndarray, min_cov: float) -> List[Tuple[int, int, float]]:
    runs: List[Tuple[int, int, float]] = []
    length = coverage.shape[0]
    i = 0
    while i < length:
        if coverage[i] <= min_cov:
            i += 1
            continue
        j = i
        total = 0.0
        while j < length and coverage[j] > min_cov:
            total += coverage[j]
            j += 1
        runs.append((i, j, total))
        i = j
    return runs


def _select_primary_run(runs: List[Tuple[int, int, float]], coverage: np.ndarray, total_len: int,
                         params: Dict) -> Tuple[int, int]:
    if not runs:
        return 0, total_len
    masses = [mass for _, _, mass in runs]
    lengths = [(end - start) for start, end, _ in runs]
    total_mass = sum(masses)

    idx = int(np.argmax(masses))
    primary_start, primary_end, _ = runs[idx]

    min_mass_ratio = float(params.get('primary_run_min_mass_ratio', 0.5))
    min_len_ratio = float(params.get('primary_run_min_length_ratio', 0.3))
    if masses[idx] < total_mass * min_mass_ratio or (primary_end - primary_start) < total_len * min_len_ratio:
        idx = int(np.argmax(lengths))
        primary_start, primary_end, _ = runs[idx]

    tighten_cov = float(params.get('tighten_min_coverage', 0.01))
    while primary_start < primary_end and coverage[primary_start] <= tighten_cov:
        primary_start += 1
    while primary_end > primary_start and coverage[primary_end - 1] <= tighten_cov:
        primary_end -= 1

    primary_start = int(max(0, min(total_len - 1, primary_start)))
    primary_end = int(max(primary_start + 1, min(total_len, primary_end)))
    return primary_start, primary_end


def _trim_axis(mask: np.ndarray, axis: int, params: Dict) -> Tuple[int, int]:
    if axis == 0:  # columns
        coverage = mask.sum(axis=0).astype(np.float32) / float(max(1, mask.shape[0]) * 255.0)
        total_len = mask.shape[1]
        limit_ratio_key = 'horizontal_trim_limit_ratio'
        limit_px_key = 'horizontal_trim_limit_px'
    else:  # rows
        coverage = mask.sum(axis=1).astype(np.float32) / float(max(1, mask.shape[1]) * 255.0)
        total_len = mask.shape[0]
        limit_ratio_key = 'vertical_trim_limit_ratio'
        limit_px_key = 'vertical_trim_limit_px'

    max_cov = coverage.max()
    if max_cov <= 0.0:
        return 0, total_len
    min_cov = max(float(params.get('run_min_coverage_abs', 0.005)),
                   float(params.get('run_min_coverage_ratio', 0.01)) * max_cov)
    runs = _find_runs(coverage, min_cov)
    limit_ratio = float(params.get(limit_ratio_key, 1.0))
    limit_px = int(params.get(limit_px_key, total_len))
    ratio_bound = total_len if limit_ratio <= 0 else int(round(total_len * min(limit_ratio, 1.0)))
    px_bound = total_len if limit_px <= 0 else limit_px
    trim_limit = max(0, min(total_len, ratio_bound, px_bound))

    if not runs:
        # only空列，允许裁掉到限制位置
        left = min(trim_limit, total_len)
        right = max(left + 1, total_len - trim_limit)
        return max(0, min(total_len - 1, left)), min(total_len, max(left + 1, right))

    tighten_cov = float(params.get('tighten_min_coverage', 0.01))

    left_idx = 0
    while left_idx < len(runs) and runs[left_idx][1] <= trim_limit:
        left_idx += 1
    if left_idx >= len(runs):
        left_idx = len(runs) - 1
    left_candidate = runs[left_idx][0]
    left = min(left_candidate, trim_limit)
    while left < total_len and coverage[left] <= tighten_cov:
        left += 1

    right_idx = len(runs) - 1
    while right_idx >= left_idx and runs[right_idx][0] >= max(0, total_len - trim_limit):
        right_idx -= 1
    if right_idx < left_idx:
        right_idx = len(runs) - 1
    right_candidate = runs[right_idx][1]
    right = max(total_len - trim_limit, right_candidate)
    if right > total_len:
        right = total_len
    while right > 0 and coverage[right - 1] <= tighten_cov:
        right -= 1

    left = int(max(0, min(total_len - 1, left)))
    right = int(max(left + 1, min(total_len, right)))
    return left, right


def _trim_edges_by_projection(mask: np.ndarray, params: Dict) -> Tuple[int, int, int, int]:
    xl, xr = _trim_axis(mask, axis=0, params=params)
    cropped = mask[:, xl:xr]
    yt, yb = _trim_axis(cropped, axis=1, params=params)
    return xl, xr, yt, yb


def trim_projection(gray: np.ndarray, params: Dict, debug: bool = False) -> Tuple[int, int, int, int]:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h
    bin_img = binarize(gray, mode=str(params.get('binarize', 'otsu')).lower(),
                       adaptive_block=int(params.get('adaptive_block', 31)),
                       adaptive_C=int(params.get('adaptive_C', 3)))
    mask = (bin_img > 0).astype(np.uint8) * 255
    xl, xr, yt, yb = _trim_edges_by_projection(mask, params)
    return xl, xr, yt, yb


def trim_projection_from_bin(bin_img: np.ndarray, params: Dict) -> Tuple[int, int, int, int]:
    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h
    mask = (bin_img > 0).astype(np.uint8) * 255
    xl, xr, yt, yb = _trim_edges_by_projection(mask, params)
    return xl, xr, yt, yb
