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
    Supports multiple iterations to completely remove thick borders.

    Args:
        binary_img: Binary image (foreground=255, background=0)
        params: Configuration parameters

    Returns:
        Tuple of (xl, xr, yt, yb) - trim coordinates
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h

    # Get iteration settings
    max_iterations = int(params.get('max_iterations', 1))

    # Initialize coordinates
    total_xl, total_xr = 0, w
    current_img = binary_img.copy()

    # Iterate border removal
    for iteration in range(max_iterations):
        # Step 1: Find border regions using horizontal projection
        left_cut, right_cut = _find_border_cuts_by_projection(current_img, params)

        # If no border found in this iteration, break
        if left_cut == 0 and right_cut == 0:
            break

        # Convert cuts to trim coordinates for this iteration
        xl = left_cut
        xr = current_img.shape[1] - right_cut

        # Ensure valid horizontal coordinates
        xl = max(0, min(current_img.shape[1]-1, xl))
        xr = max(xl+1, min(current_img.shape[1], xr))

        # Update total coordinates (accumulate cuts)
        total_xl += xl
        total_xr = total_xl + (xr - xl)

        # Prepare for next iteration - crop the current image
        if xl < xr and iteration < max_iterations - 1:
            current_img = current_img[:, xl:xr]
        else:
            break

    # Step 2: Perform vertical projection analysis on the final horizontally trimmed region
    if total_xl < total_xr:
        final_region = binary_img[:, total_xl:total_xr]
        yt, yb = _find_vertical_trim_bounds(final_region, params)
    else:
        yt, yb = 0, h

    # Ensure valid coordinates
    total_xl = max(0, min(w-1, total_xl))
    total_xr = max(total_xl+1, min(w, total_xr))
    yt = max(0, min(h-1, yt))
    yb = max(yt+1, min(h, yb))

    return total_xl, total_xr, yt, yb


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

    # Calculate horizontal projection (column-wise coverage)
    horizontal_projection = binary_img.sum(axis=0).astype(np.float32) / (h * 255.0)

    max_coverage = horizontal_projection.max() if horizontal_projection.size > 0 else 0.0
    if max_coverage <= 0.0:
        return 0, 0

    max_border_width = int(w * max_border_width_ratio)

    left_cut = 0
    right_cut = 0

    # Check left edge for border using spike detection
    left_projection = horizontal_projection[:max_border_width]
    left_debug = _debug_border_detection(left_projection, params, is_left=True, total_width=w, global_max_coverage=max_coverage)
    left_cut = left_debug.get('final_cut_pos', 0)

    # Check right edge for border using spike detection
    right_projection = horizontal_projection[-max_border_width:]
    right_debug = _debug_border_detection(right_projection, params, is_left=False, total_width=w, global_max_coverage=max_coverage)
    right_cut = right_debug.get('final_cut_pos', 0)

    return left_cut, right_cut


def _debug_border_detection(projection: np.ndarray, params: Dict, is_left: bool, total_width: int = None, global_max_coverage: float = None) -> Dict:
    """
    Debug function to analyze border detection process.

    Returns detailed information about detection process for debugging.
    """
    if projection.size == 0:
        return {'error': 'Empty projection'}

    max_coverage = projection.max() if projection.size > 0 else 0.0
    if max_coverage <= 0.0:
        return {'error': 'No coverage data'}

    # Use global max coverage for threshold calculation if provided (to match main algorithm)
    threshold_base = global_max_coverage if global_max_coverage is not None else max_coverage

    # Calculate spike lengths based on total width if provided, otherwise use projection length
    if total_width is not None:
        spike_min_length = max(1, int(total_width * float(params.get('spike_min_length_ratio', 0.02))))
        spike_max_length = max(spike_min_length, int(total_width * float(params.get('spike_max_length_ratio', 0.1))))
    else:
        spike_min_length = max(1, int(len(projection) * float(params.get('spike_min_length_ratio', 0.02))))
        spike_max_length = max(spike_min_length, int(len(projection) * float(params.get('spike_max_length_ratio', 0.1))))

    gradient_threshold = threshold_base * float(params.get('spike_gradient_threshold', 0.4))
    prominence_ratio = float(params.get('spike_prominence_ratio', 0.6))
    border_threshold = threshold_base * float(params.get('border_threshold_ratio', 0.5))
    edge_tolerance = int(params.get('edge_tolerance', 8))

    debug_info = {
        'projection_length': len(projection),
        'max_coverage': max_coverage,
        'border_threshold': border_threshold,
        'spike_min_length': spike_min_length,
        'spike_max_length': spike_max_length,
        'gradient_threshold': gradient_threshold,
        'prominence_ratio': prominence_ratio,
        'edge_tolerance': edge_tolerance,
        'is_left': is_left,
        'detection_attempts': [],
        'final_cut_pos': 0,
        'max_border_length': 0
    }

    best_cut_pos = 0
    max_border_length = 0

    if is_left:
        border_start = -1
        for i in range(len(projection)):
            if projection[i] >= border_threshold:
                if border_start == -1:
                    border_start = i
            else:
                if border_start != -1:
                    border_length = i - border_start
                    border_avg = projection[border_start:i].mean()

                    # Check if border is close enough to edge
                    is_near_edge = border_start <= edge_tolerance

                    lookahead_end = min(len(projection), i + border_length)
                    content_avg = projection[i:lookahead_end].mean() if i < len(projection) else 0
                    drop_magnitude = border_avg - content_avg

                    gradient_ok = drop_magnitude >= gradient_threshold
                    prominence_ok = drop_magnitude >= border_avg * prominence_ratio
                    length_ok = spike_min_length <= border_length <= spike_max_length

                    attempt = {
                        'position': i,
                        'border_start': border_start,
                        'border_length': border_length,
                        'border_avg': border_avg,
                        'content_avg': content_avg,
                        'drop_magnitude': drop_magnitude,
                        'is_near_edge': is_near_edge,
                        'length_ok': length_ok,
                        'gradient_ok': gradient_ok,
                        'prominence_ok': prominence_ok,
                        'passed': is_near_edge and length_ok and gradient_ok and prominence_ok
                    }
                    # Add detailed projection values for debugging
                    attempt['projection_values'] = projection[border_start:i].tolist() if border_length > 0 else []
                    attempt['gradient_threshold_check'] = gradient_threshold
                    attempt['prominence_threshold_check'] = border_avg * prominence_ratio if border_length > 0 else 0

                    debug_info['detection_attempts'].append(attempt)

                    if attempt['passed'] and border_length > max_border_length:
                        max_border_length = border_length
                        best_cut_pos = i

                    border_start = -1
    else:
        border_start = -1
        for i in range(len(projection) - 1, -1, -1):
            if projection[i] >= border_threshold:
                if border_start == -1:
                    border_start = i
            else:
                if border_start != -1:
                    border_length = border_start - i
                    border_avg = projection[i+1:border_start+1].mean()

                    # Check if border is close enough to edge
                    distance_from_edge = len(projection) - 1 - border_start
                    is_near_edge = distance_from_edge <= edge_tolerance

                    lookahead_start = max(0, i - border_length + 1)
                    content_avg = projection[lookahead_start:i+1].mean() if i >= 0 else 0
                    drop_magnitude = border_avg - content_avg

                    gradient_ok = drop_magnitude >= gradient_threshold
                    prominence_ok = drop_magnitude >= border_avg * prominence_ratio
                    length_ok = spike_min_length <= border_length <= spike_max_length

                    attempt = {
                        'position': i,
                        'border_start': border_start,
                        'border_length': border_length,
                        'border_avg': border_avg,
                        'content_avg': content_avg,
                        'drop_magnitude': drop_magnitude,
                        'is_near_edge': is_near_edge,
                        'length_ok': length_ok,
                        'gradient_ok': gradient_ok,
                        'prominence_ok': prominence_ok,
                        'passed': is_near_edge and length_ok and gradient_ok and prominence_ok
                    }
                    # Add detailed projection values for debugging
                    attempt['projection_values'] = projection[i+1:border_start+1].tolist() if border_length > 0 else []
                    attempt['gradient_threshold_check'] = gradient_threshold
                    attempt['prominence_threshold_check'] = border_avg * prominence_ratio if border_length > 0 else 0

                    debug_info['detection_attempts'].append(attempt)

                    if attempt['passed'] and border_length > max_border_length:
                        max_border_length = border_length
                        best_cut_pos = len(projection) - border_start

                    border_start = -1

    debug_info['final_cut_pos'] = best_cut_pos
    debug_info['max_border_length'] = max_border_length
    return debug_info


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