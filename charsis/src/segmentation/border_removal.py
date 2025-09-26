"""
Border removal for ancient text images using horizontal projection analysis.

Remove adhesion borders and frames that may still remain after projection trimming.
Uses horizontal projection to detect short, high-intensity regions at the edges
that typically indicate border structures.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import cv2


def trim_border_from_bin(binary_img: np.ndarray, params: Dict) -> Tuple[int, int, int, int]:
    """
    Find border trim coordinates using horizontal and vertical projection analysis.

    This function first removes horizontal borders, then performs vertical
    projection analysis to tighten the top/bottom boundaries.

    Args:
        binary_img: Binary image (foreground=255, background=0)
        params: Configuration parameters

    Returns:
        Tuple of (xl, xr, yt, yb) - trim coordinates
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h

    # Step 1: Find border regions using horizontal projection
    left_cut, right_cut = _find_border_cuts_by_projection(binary_img, params)

    # Convert cuts to trim coordinates
    xl = left_cut
    xr = w - right_cut

    # Ensure valid horizontal coordinates
    xl = max(0, min(w-1, xl))
    xr = max(xl+1, min(w, xr))

    # Step 2: Perform vertical projection analysis on the horizontally trimmed region
    if xl < xr:
        trimmed_region = binary_img[:, xl:xr]
        yt, yb = _find_vertical_trim_bounds(trimmed_region, params)
    else:
        yt, yb = 0, h

    # Ensure valid vertical coordinates
    yt = max(0, min(h-1, yt))
    yb = max(yt+1, min(h, yb))

    return xl, xr, yt, yb


def _find_border_cuts_by_projection(binary_img: np.ndarray, params: Dict) -> Tuple[int, int]:
    """
    Find border cut positions using horizontal projection analysis.

    Args:
        binary_img: Binary image (foreground=255)
        params: Configuration parameters

    Returns:
        Tuple of (left_cut_pixels, right_cut_pixels)
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, 0

    # Parameters
    max_border_width_ratio = float(params.get('border_max_width_ratio', 0.2))    # 最大边框宽度占比
    min_height_ratio = float(params.get('border_min_height_ratio', 0.6))         # 边框最小高度占比
    intensity_threshold_ratio = float(params.get('border_intensity_ratio', 0.3)) # 强度阈值占比

    # Calculate horizontal projection (column-wise coverage)
    horizontal_projection = binary_img.sum(axis=0).astype(np.float32) / (h * 255.0)

    max_coverage = horizontal_projection.max() if horizontal_projection.size > 0 else 0.0
    if max_coverage <= 0.0:
        return 0, 0

    # Intensity threshold for detecting high-coverage regions
    intensity_threshold = max_coverage * intensity_threshold_ratio
    min_height_threshold = min_height_ratio  # Coverage ratio threshold

    max_border_width = int(w * max_border_width_ratio)

    left_cut = 0
    right_cut = 0

    # Check left edge for border
    left_cut = _find_edge_border_cut(
        horizontal_projection[:max_border_width],
        intensity_threshold,
        min_height_threshold,
        is_left=True
    )

    # Check right edge for border
    right_cut = _find_edge_border_cut(
        horizontal_projection[-max_border_width:],
        intensity_threshold,
        min_height_threshold,
        is_left=False
    )

    return left_cut, right_cut


def _find_edge_border_cut(projection: np.ndarray, intensity_threshold: float,
                         min_height_threshold: float, is_left: bool) -> int:
    """
    Find border cut position for one edge.

    Args:
        projection: Horizontal projection for the edge region
        intensity_threshold: Minimum intensity to consider as border
        min_height_threshold: Minimum height ratio for border
        is_left: Whether this is left edge (True) or right edge (False)

    Returns:
        Number of pixels to cut from this edge
    """
    if projection.size == 0:
        return 0

    # Find regions with sufficient coverage (potential borders)
    high_coverage_mask = projection >= intensity_threshold

    if not np.any(high_coverage_mask):
        return 0

    if is_left:
        # For left edge, find the rightmost position of continuous high coverage from left
        cut_pos = 0
        consecutive_high = 0
        max_consecutive = 0
        best_cut_pos = 0

        for i in range(len(projection)):
            if high_coverage_mask[i]:
                consecutive_high += 1
                cut_pos = i + 1
            else:
                if consecutive_high > max_consecutive:
                    max_consecutive = consecutive_high
                    best_cut_pos = cut_pos
                consecutive_high = 0

        # Check final consecutive region
        if consecutive_high > max_consecutive:
            best_cut_pos = cut_pos

        # Validate the cut region meets height requirement
        if best_cut_pos > 0:
            border_height = np.mean(projection[:best_cut_pos])
            if border_height >= min_height_threshold:
                return best_cut_pos

    else:
        # For right edge, find the leftmost position of continuous high coverage from right
        cut_pos = 0
        consecutive_high = 0
        max_consecutive = 0
        best_cut_pos = 0

        for i in range(len(projection) - 1, -1, -1):
            if high_coverage_mask[i]:
                consecutive_high += 1
                cut_pos = len(projection) - i
            else:
                if consecutive_high > max_consecutive:
                    max_consecutive = consecutive_high
                    best_cut_pos = cut_pos
                consecutive_high = 0

        # Check final consecutive region
        if consecutive_high > max_consecutive:
            best_cut_pos = cut_pos

        # Validate the cut region meets height requirement
        if best_cut_pos > 0:
            start_pos = len(projection) - best_cut_pos
            border_height = np.mean(projection[start_pos:])
            if border_height >= min_height_threshold:
                return best_cut_pos

    return 0


def _find_vertical_trim_bounds(binary_img: np.ndarray, params: Dict) -> Tuple[int, int]:
    """
    Find vertical trim bounds by removing only white margins.

    This function removes pure white (empty) rows from top and bottom,
    but does not tighten into text content.

    Args:
        binary_img: Binary image after horizontal border removal
        params: Configuration parameters

    Returns:
        Tuple of (yt, yb) - vertical trim coordinates
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, h

    # Calculate vertical projection (row-wise coverage)
    vertical_projection = binary_img.sum(axis=1).astype(np.float32) / (w * 255.0)

    if vertical_projection.size == 0:
        return 0, h

    # Parameters for white margin removal
    white_threshold = float(params.get('vertical_white_threshold', 0.001))  # Nearly zero content

    # Find first and last rows with any meaningful content
    content_rows = vertical_projection > white_threshold
    if not np.any(content_rows):
        return 0, h

    # Find first and last content rows (remove only pure white margins)
    content_indices = np.where(content_rows)[0]
    yt = int(content_indices[0])
    yb = int(content_indices[-1] + 1)

    # Ensure valid bounds
    yt = max(0, min(h-1, yt))
    yb = max(yt+1, min(h, yb))

    return yt, yb


def analyze_border_removal_effect(original: np.ndarray, cleaned: np.ndarray) -> Dict:
    """
    Analyze the effect of border removal for debugging.

    Returns:
        Dictionary with analysis results
    """
    if original is None or cleaned is None:
        return {'error': 'Invalid input images'}

    # Convert to binary for analysis
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original.copy()

    if len(cleaned.shape) == 3:
        clean_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    else:
        clean_gray = cleaned.copy()

    # Binarize
    _, orig_bin = cv2.threshold(orig_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, clean_bin = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate differences
    removed_pixels = cv2.bitwise_and(orig_bin, cv2.bitwise_not(clean_bin))
    removed_area = np.sum(removed_pixels > 0)
    total_area = np.sum(orig_bin > 0)

    return {
        'removed_area': int(removed_area),
        'total_area': int(total_area),
        'removal_ratio': float(removed_area / max(1, total_area)),
        'has_effect': removed_area > 0
    }