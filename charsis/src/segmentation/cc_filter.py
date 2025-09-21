"""
Connected-component pre-filter for segmentation.

Goal: remove clearly abnormal regions before projection trimming, with
minimal parameters and conservative rules:
  - tiny components relative to ROI area and touching image edges
  - edge-touching, extremely slender components (very high aspect ratio)
  - edge-touching components with min dimension below a small pixel threshold

Operates in-place on a binary (uint8) image with foreground=255.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import cv2


def refine_binary_components(bin_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Remove abnormal connected components from a binary image in-place, and
    return the cleaned image.

    Parameters (with conservative defaults provided by config):
      - min_area_ratio: components with area < ratio * (H*W) AND touching edge are removed
      - edge_margin: how close to edge counts as touching (in pixels)
      - max_aspect_for_edge: if component touches edge and max(w/h, h/w) >= this, remove
      - min_dim_px: if component touches edge and min(w, h) <= this, remove
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img
    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return bin_img

    min_area_ratio = float(params.get('min_area_ratio', 0.0015))
    edge_margin = int(params.get('edge_margin', 6))
    max_aspect_for_edge = float(params.get('max_aspect_for_edge', 10.0))
    min_dim_px = int(params.get('min_dim_px', 2))

    area_thr = max(1, int(min_area_ratio * h * w))
    m = (bin_img > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return bin_img
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        # edge contact test (with margin)
        touches_edge = (
            x <= edge_margin or y <= edge_margin or
            (x + ww) >= (w - edge_margin) or (y + hh) >= (h - edge_margin)
        )
        if not touches_edge:
            continue
        # rule 1: tiny and edge-touching
        if area < area_thr:
            bin_img[labels == i] = 0
            continue
        # rule 2: extremely slender at edge
        aspect = float(max(ww / max(1, hh), hh / max(1, ww)))
        if aspect >= max_aspect_for_edge:
            bin_img[labels == i] = 0
            continue
        # rule 3: min dimension too small at edge
        if min(ww, hh) <= min_dim_px:
            bin_img[labels == i] = 0
            continue
    return bin_img
