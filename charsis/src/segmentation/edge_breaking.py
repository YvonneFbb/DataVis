"""
Edge adhesion breaking for ancient text segmentation.

Intelligently detect and break thin connections at image edges that typically
represent adhesion to borders or neighboring characters, while preserving
legitimate character strokes.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import cv2


def intelligent_edge_breaking(bin_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Intelligently break thin edge connections while preserving character integrity.

    Strategy:
    1. Only process edge-touching connected components
    2. Analyze geometric features to identify adhesions
    3. Apply gentle morphological opening with small kernels
    4. Verify breaking results to avoid over-breaking

    Args:
        bin_img: Binary image (uint8, foreground=255)
        params: Configuration parameters

    Returns:
        Cleaned binary image with adhesions broken
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img

    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return bin_img

    # Extract parameters
    min_aspect_ratio = float(params.get('min_aspect_ratio', 3.0))
    kernel_size = int(params.get('kernel_size', 2))
    min_remaining_area = int(params.get('min_remaining_area', 10))
    edge_margin = int(params.get('edge_margin', 2))

    # Connected component analysis
    m = (bin_img > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num <= 1:
        return bin_img

    result = bin_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # 1. Check if component touches image edges (with margin)
        touches_edge = (
            x <= edge_margin or y <= edge_margin or
            (x + ww) >= (w - edge_margin) or (y + hh) >= (h - edge_margin)
        )

        if not touches_edge:
            continue

        # 2. Analyze geometric features for adhesion detection
        aspect_ratio = float(max(ww / max(hh, 1), hh / max(ww, 1)))

        # Skip if not elongated enough (likely not adhesion)
        if aspect_ratio < min_aspect_ratio:
            continue

        # 3. Extract component mask
        component_mask = (labels == i).astype(np.uint8) * 255

        # 4. Apply gentle morphological opening
        opened = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, kernel)

        # 5. Verify breaking results
        new_num, new_labels, new_stats, _ = cv2.connectedComponentsWithStats(
            (opened > 0).astype(np.uint8), connectivity=8
        )

        if new_num > 1:  # Component was broken
            # Count valid remaining components
            valid_components = sum(
                1 for j in range(1, new_num)
                if new_stats[j][4] >= min_remaining_area
            )

            # Apply breaking only if we have valid components remaining
            if valid_components >= 1:
                # Remove original component and add broken parts
                result[labels == i] = 0
                result[opened > 0] = 255

    return result


def selective_edge_breaking(bin_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Alternative approach: only apply breaking in edge regions.

    This is more conservative and only processes pixels near image boundaries.
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img

    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return bin_img

    edge_ratio = float(params.get('edge_ratio', 0.15))
    kernel_size = int(params.get('kernel_size', 2))

    edge_w = max(1, int(w * edge_ratio))
    edge_h = max(1, int(h * edge_ratio))

    result = bin_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Define edge regions and apply breaking
    regions = [
        (slice(None), slice(None, edge_w)),           # Left edge
        (slice(None), slice(-edge_w, None)),          # Right edge
        (slice(None, edge_h), slice(None)),           # Top edge
        (slice(-edge_h, None), slice(None)),          # Bottom edge
    ]

    for row_slice, col_slice in regions:
        region = result[row_slice, col_slice]
        if region.size > 0:
            opened = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
            result[row_slice, col_slice] = opened

    return result


def break_edge_adhesions(bin_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Main entry point for edge adhesion breaking.

    Chooses between intelligent or selective approach based on configuration.
    """
    method = str(params.get('method', 'intelligent')).lower()

    if method == 'selective':
        return selective_edge_breaking(bin_img, params)
    else:
        return intelligent_edge_breaking(bin_img, params)