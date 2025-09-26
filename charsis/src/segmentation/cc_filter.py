"""
Connected-component pre-filter for segmentation.

Goal: remove clearly abnormal regions before projection trimming, with
minimal parameters and conservative rules:
  - tiny components relative to ROI area and touching the image border directly
  - tiny components relative to ROI area and within an edge margin band
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
    Remove abnormal connected components from a binary image based on 3-tier classification:
    1. Border touching (actual contact): components touching the exact image border
    2. Edge zone (margin area): components within edge margin but not touching border
    3. Interior (other): components not in edge zone

    Each tier has its own area ratio threshold for removal.

    Parameters:
      - border_touch_margin: pixel range for border touching detection (default: 0)
      - edge_zone_margin: pixel range for edge zone detection (default: 6)
      - border_touch_min_area_ratio: area ratio threshold for border touching components
      - edge_zone_min_area_ratio: area ratio threshold for edge zone components
      - interior_min_area_ratio: area ratio threshold for interior components
      - max_aspect_for_edge: max aspect ratio allowed for edge/border components
      - min_dim_px: min dimension allowed for edge/border components
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img
    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return bin_img

    # Parameters for 3-tier classification
    border_touch_margin = int(params.get('border_touch_margin', 0))           # 实际触边范围
    edge_zone_margin = int(params.get('edge_zone_margin', 6))                 # 边缘区域范围
    border_touch_min_area_ratio = float(params.get('border_touch_min_area_ratio', 0.001))  # 触边组件面积阈值
    edge_zone_min_area_ratio = float(params.get('edge_zone_min_area_ratio', 0.002))        # 边缘组件面积阈值
    interior_min_area_ratio = float(params.get('interior_min_area_ratio', 0.0005))         # 内部组件面积阈值

    # Shape constraints for edge/border components
    max_aspect_for_edge = float(params.get('max_aspect_for_edge', 10.0))
    min_dim_px = int(params.get('min_dim_px', 2))

    # Calculate area thresholds
    total_area = h * w
    border_area_thr = max(1, int(border_touch_min_area_ratio * total_area))
    edge_area_thr = max(1, int(edge_zone_min_area_ratio * total_area))
    interior_area_thr = max(1, int(interior_min_area_ratio * total_area))

    m = (bin_img > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return bin_img

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Classify component based on position
        component_type, should_remove = _classify_and_filter_component(
            x, y, ww, hh, area, w, h,
            border_touch_margin, edge_zone_margin,
            border_area_thr, edge_area_thr, interior_area_thr,
            max_aspect_for_edge, min_dim_px
        )

        if should_remove:
            bin_img[labels == i] = 0

    return bin_img


def _classify_and_filter_component(x: int, y: int, ww: int, hh: int, area: int,
                                   img_w: int, img_h: int,
                                   border_margin: int, edge_margin: int,
                                   border_thr: int, edge_thr: int, interior_thr: int,
                                   max_aspect: float, min_dim: int) -> tuple[str, bool]:
    """
    Classify component into border-touching, edge-zone, or interior, and decide if it should be removed.

    Returns:
        (component_type, should_remove): component classification and removal decision
    """

    # Check if component touches actual border (within border_margin)
    touches_border = (
        x <= border_margin or y <= border_margin or
        (x + ww) >= (img_w - border_margin) or (y + hh) >= (img_h - border_margin)
    )

    if touches_border:
        component_type = "border_touch"
        # Apply border touching rules
        if area < border_thr:
            return component_type, True

        # Shape constraints for border components
        aspect = max(ww / max(1, hh), hh / max(1, ww))
        if aspect >= max_aspect or min(ww, hh) <= min_dim:
            return component_type, True

        return component_type, False

    # Check if component is in edge zone (within edge_margin but not touching border)
    in_edge_zone = (
        x <= edge_margin or y <= edge_margin or
        (x + ww) >= (img_w - edge_margin) or (y + hh) >= (img_h - edge_margin)
    )

    if in_edge_zone:
        component_type = "edge_zone"
        # Apply edge zone rules
        if area < edge_thr:
            return component_type, True

        # Shape constraints for edge components
        aspect = max(ww / max(1, hh), hh / max(1, ww))
        if aspect >= max_aspect or min(ww, hh) <= min_dim:
            return component_type, True

        return component_type, False

    # Interior component
    component_type = "interior"
    # Apply interior rules (usually more lenient)
    if area < interior_thr:
        return component_type, True

    return component_type, False
