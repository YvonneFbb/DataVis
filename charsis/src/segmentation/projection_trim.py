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
    else:  # rows
        coverage = mask.sum(axis=1).astype(np.float32) / float(max(1, mask.shape[1]) * 255.0)
        total_len = mask.shape[0]

    max_cov = coverage.max()
    if max_cov <= 0.0:
        return 0, total_len
    min_cov = max(float(params.get('run_min_coverage_abs', 0.005)),
                   float(params.get('run_min_coverage_ratio', 0.01)) * max_cov)
    runs = _find_runs(coverage, min_cov)
    return _select_primary_run(runs, coverage, total_len, params)


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
