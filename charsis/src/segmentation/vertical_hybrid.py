"""
ocrmac-driven segmentation (CC + projection trimming, simplified).

LiveText detection → connected-component filtering → projection trimming
(src/segmentation/projection_trim.py) → save crops and debug. Legacy
projection/grabcut/morph code moved to vertical_hybrid_legacy.py and is not
used by default.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import cv2
import numpy as np

# Lazy import ocrmac (Apple LiveText)
try:
    from ocrmac import ocrmac as _ocrmac
except ImportError as _e:
    _ocrmac = None
    _OCRMAC_IMPORT_ERROR = str(_e)
else:
    _OCRMAC_IMPORT_ERROR = None

try:
    from src.config import (
        PREOCR_DIR, SEGMENTS_DIR,
        SEGMENT_REFINE_CONFIG, PROJECTION_TRIM_CONFIG, CC_FILTER_CONFIG,
        BORDER_REMOVAL_CONFIG, NOISE_REMOVAL_CONFIG,
    )
except Exception as e:
    raise RuntimeError(
        f"无法导入必要配置 (PREOCR_DIR / SEGMENTS_DIR / SEGMENT_REFINE_CONFIG / PROJECTION_TRIM_CONFIG)。原始错误: {e}"
    )

from src.segmentation.projection_trim import trim_projection_from_bin, binarize
from src.segmentation.cc_filter import refine_binary_components
from src.segmentation.noise_removal import remove_noise_patches
from src.segmentation.border_removal import trim_border_from_bin, _debug_border_detection


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _create_border_projection_viz(border_region: np.ndarray, params: Dict, max_height: int = 120,
                                   xl_cut: int = 0, xr_cut: int = 0, target_width: int = 400) -> np.ndarray:
    """
    Create border projection visualization that can be embedded in main debug image.

    Args:
        border_region: Binary region after projection trimming
        params: Border removal configuration
        max_height: Maximum height for the visualization
        xl_cut: Left border cut position (relative to border_region)
        xr_cut: Right border cut position (relative to border_region)
        target_width: Target width for the visualization

    Returns:
        Visualization image showing projection, detection zones, and cut positions
    """
    if border_region.size == 0:
        return np.full((max_height, target_width, 3), 255, dtype=np.uint8)

    # Calculate projections
    border_hproj = border_region.sum(axis=0).astype(np.float32)
    if border_hproj.size == 0 or border_hproj.max() == 0:
        return np.full((max_height, target_width, 3), 255, dtype=np.uint8)

    bhp_coverage = border_hproj / (border_region.shape[0] * 255.0)
    max_coverage = bhp_coverage.max()

    # Get parameters
    max_width = int(border_region.shape[1] * params.get('border_max_width_ratio', 0.2))
    border_threshold = max_coverage * params.get('border_threshold_ratio', 0.5)

    # Create visualization
    viz_width = target_width  # Use the specified target width
    viz_height = max_height
    viz_img = np.full((viz_height, viz_width, 3), 255, dtype=np.uint8)

    # Scale projection to fit
    if len(bhp_coverage) > 0:
        scale_x = viz_width / len(bhp_coverage)
        proj_height = int(viz_height * 0.8)  # Leave space for labels

        # Draw projection bars
        for i, val in enumerate(bhp_coverage):
            x = int(i * scale_x)
            bar_height = int(val / max_coverage * proj_height) if max_coverage > 0 else 0
            y_start = viz_height - 20 - bar_height
            y_end = viz_height - 20

            if bar_height > 0:
                cv2.rectangle(viz_img, (x, y_start), (min(x+int(scale_x)+1, viz_width-1), y_end), (128, 128, 128), -1)

        # Draw detection zones
        left_zone_end = int(max_width * scale_x)
        right_zone_start = viz_width - int(max_width * scale_x)

        # Left zone
        cv2.rectangle(viz_img, (0, 10), (left_zone_end, 30), (255, 0, 0), 2)
        cv2.putText(viz_img, "LEFT ZONE", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Right zone
        if right_zone_start > left_zone_end:
            cv2.rectangle(viz_img, (right_zone_start, 10), (viz_width-1, 30), (255, 0, 0), 2)
            cv2.putText(viz_img, "RIGHT ZONE", (right_zone_start+5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw threshold line
        threshold_y = viz_height - 20 - int(border_threshold / max_coverage * proj_height) if max_coverage > 0 else viz_height - 20
        cv2.line(viz_img, (0, threshold_y), (viz_width-1, threshold_y), (0, 255, 0), 2)
        cv2.putText(viz_img, f"Threshold: {border_threshold:.3f}", (5, threshold_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Draw actual border cut positions
        if xl_cut > 0:
            xl_viz = int(xl_cut * scale_x)
            cv2.line(viz_img, (xl_viz, 35), (xl_viz, viz_height-25), (0, 0, 255), 2)
            cv2.putText(viz_img, f"L:{xl_cut}", (xl_viz+2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        if xr_cut < border_region.shape[1]:
            xr_viz = int(xr_cut * scale_x)
            cv2.line(viz_img, (xr_viz, 35), (xr_viz, viz_height-25), (0, 0, 255), 2)
            cv2.putText(viz_img, f"R:{xr_cut}", (max(0, xr_viz-25), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    return viz_img


def _create_border_debug_image(border_region: np.ndarray,
                              xl_border: int, xr_border: int,
                              xl_proj: int, xr_proj: int,
                              params: Dict) -> np.ndarray:
    """
    Create a dedicated border detection debug image.

    Args:
        border_region: The binary region after projection trimming
        xl_border, xr_border: Border detection results
        xl_proj, xr_proj: Projection trimming results
        params: Border removal configuration

    Returns:
        Debug image showing border detection process
    """
    if border_region.size == 0:
        return np.full((400, 800, 3), 255, dtype=np.uint8)

    debug_width = 800
    debug_height = 800  # Increase height to fit all content
    debug_img = np.full((debug_height, debug_width, 3), 255, dtype=np.uint8)

    # Calculate projections
    border_hproj = border_region.sum(axis=0).astype(np.float32)
    if border_hproj.size == 0 or border_hproj.max() == 0:
        cv2.putText(debug_img, "No projection data", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return debug_img

    bhp_coverage = border_hproj / (border_region.shape[0] * 255.0)  # Actual coverage ratio
    max_coverage = bhp_coverage.max()

    # Get debug info for both edges
    max_width = int(border_region.shape[1] * params.get('border_max_width_ratio', 0.2))
    debug_left = {}
    debug_right = {}

    if max_width > 0:
        # Generate debug info using the same projection as main algorithm would use
        # Main algorithm calculates projection from the actual binary image being processed
        actual_hproj = border_region.sum(axis=0).astype(np.float32) / (border_region.shape[0] * 255.0)
        actual_max_coverage = actual_hproj.max() if actual_hproj.size > 0 else 0.0

        left_proj = actual_hproj[:max_width]
        right_proj = actual_hproj[-max_width:]
        debug_left = _debug_border_detection(left_proj, params, is_left=True, total_width=border_region.shape[1], global_max_coverage=actual_max_coverage)
        debug_right = _debug_border_detection(right_proj, params, is_left=False, total_width=border_region.shape[1], global_max_coverage=actual_max_coverage)

        # Store projection info for later display
        left_max = left_proj.max() if len(left_proj) > 0 else 0
        right_max = right_proj.max() if len(right_proj) > 0 else 0

    # Draw projection histogram (top section)
    hist_height = 200
    hist_start_y = 50
    hist_width = min(700, border_region.shape[1] * 2)  # Scale for visibility

    # Draw projection bars
    for i in range(border_region.shape[1]):
        if i < len(bhp_coverage):
            bar_height = int(bhp_coverage[i] * hist_height)
            x_pos = 50 + int(i * hist_width / border_region.shape[1])
            if bar_height > 0:
                cv2.rectangle(debug_img,
                            (x_pos, hist_start_y + hist_height - bar_height),
                            (x_pos + 1, hist_start_y + hist_height),
                            (100, 100, 100), -1)

    # Mark detection zones
    left_zone_end = 50 + int(max_width * hist_width / border_region.shape[1])
    right_zone_start = 50 + int((border_region.shape[1] - max_width) * hist_width / border_region.shape[1])

    # Left detection zone
    cv2.rectangle(debug_img, (50, hist_start_y), (left_zone_end, hist_start_y + hist_height), (255, 200, 200), 2)
    cv2.putText(debug_img, "LEFT ZONE", (55, hist_start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Right detection zone
    cv2.rectangle(debug_img, (right_zone_start, hist_start_y), (50 + hist_width, hist_start_y + hist_height), (200, 200, 255), 2)
    cv2.putText(debug_img, "RIGHT ZONE", (right_zone_start + 5, hist_start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Mark border threshold line
    threshold_y = hist_start_y + hist_height - int(max_coverage * 0.5 * hist_height)
    cv2.line(debug_img, (50, threshold_y), (50 + hist_width, threshold_y), (0, 255, 0), 1)
    cv2.putText(debug_img, f"Border Threshold: {max_coverage * 0.5:.3f}",
                (50 + hist_width + 10, threshold_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Mark cut positions
    left_cut_rel = xl_border - xl_proj
    right_cut_rel = xr_border - xl_proj

    if left_cut_rel > 0:
        cut_x = 50 + int(left_cut_rel * hist_width / border_region.shape[1])
        cv2.line(debug_img, (cut_x, hist_start_y), (cut_x, hist_start_y + hist_height), (255, 0, 0), 2)
        cv2.putText(debug_img, f"L_CUT: {left_cut_rel}", (cut_x + 5, hist_start_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    if right_cut_rel < border_region.shape[1]:
        cut_x = 50 + int(right_cut_rel * hist_width / border_region.shape[1])
        cv2.line(debug_img, (cut_x, hist_start_y), (cut_x, hist_start_y + hist_height), (0, 0, 255), 2)
        cv2.putText(debug_img, f"R_CUT: {border_region.shape[1] - right_cut_rel}", (cut_x - 60, hist_start_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Display detection parameters (bottom section)
    text_start_y = 300
    line_height = 25

    cv2.putText(debug_img, "BORDER DETECTION PARAMETERS:", (20, text_start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    param_texts = [
        f"Region Width: {border_region.shape[1]}px, Detection Zone: {max_width}px ({params.get('border_max_width_ratio', 0.2)*100:.1f}%)",
        f"Max Coverage: {max_coverage:.4f}",
        f"Border Threshold (50%): {max_coverage * 0.5:.4f}",
        f"Spike Min Length: {params.get('spike_min_length_ratio', 0.02)*100:.1f}% * {border_region.shape[1]}px (total) = {int(border_region.shape[1] * params.get('spike_min_length_ratio', 0.02))}px",
        f"Spike Max Length: {params.get('spike_max_length_ratio', 0.1)*100:.1f}% * {border_region.shape[1]}px (total) = {int(border_region.shape[1] * params.get('spike_max_length_ratio', 0.1))}px",
        f"Gradient Threshold: {params.get('spike_gradient_threshold', 0.2)*100:.1f}% * {max_coverage:.3f} = {max_coverage * params.get('spike_gradient_threshold', 0.2):.4f}",
        f"Prominence Ratio: {params.get('spike_prominence_ratio', 0.2)*100:.1f}%"
    ]

    for i, text in enumerate(param_texts):
        cv2.putText(debug_img, text, (20, text_start_y + 30 + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display detection results
    result_start_y = text_start_y + 180  # Reduce gap
    cv2.putText(debug_img, "DETECTION RESULTS:", (20, result_start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Left side results
    left_y = result_start_y + 30
    if debug_left.get('detection_attempts'):
        cv2.putText(debug_img, f"LEFT: {len(debug_left['detection_attempts'])} attempts, cut at {debug_left['final_cut_pos']}",
                    (20, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for i, attempt in enumerate(debug_left['detection_attempts'][:2]):  # Show first 2 attempts with details
            status = "PASS" if attempt['passed'] else "FAIL"
            detail_text = f"  Attempt {i+1}: pos={attempt['position']}, len={attempt['border_length']}, drop={attempt['drop_magnitude']:.4f} - {status}"
            cv2.putText(debug_img, detail_text, (30, left_y + 20 + i * 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 0, 0), 1)

            # Show detailed condition breakdown
            proj_max = max(attempt.get('projection_values', [0])) if attempt.get('projection_values') else 0
            conditions = [
                f"    proj_max: {proj_max:.3f} >= border_thresh: {debug_left.get('border_threshold', 0):.3f}",
                f"    near_edge: {attempt.get('is_near_edge', False)} (start: {attempt.get('border_start', 0)})",
                f"    len: {attempt.get('length_ok', False)} grad: {attempt.get('gradient_ok', False)} prom: {attempt.get('prominence_ok', False)}"
            ]
            for j, cond in enumerate(conditions):
                cv2.putText(debug_img, cond, (30, left_y + 35 + i * 80 + j * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 0, 0), 1)

        left_bottom = left_y + 20 + len(debug_left.get('detection_attempts', [])) * 80
    else:
        cv2.putText(debug_img, "LEFT: No attempts", (20, left_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        left_bottom = left_y

    # Right side results - start after left results with spacing
    right_y = max(left_bottom + 40, result_start_y + 100)
    cv2.putText(debug_img, "RIGHT DEBUG STATUS:", (20, right_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    right_detail_y = right_y + 25
    if debug_right:
        if 'error' in debug_right:
            debug_status = f"Error: {debug_right['error']}"
            cv2.putText(debug_img, debug_status, (20, right_detail_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            right_detail_y += 25
        elif debug_right.get('detection_attempts'):
            cv2.putText(debug_img, f"{len(debug_right['detection_attempts'])} attempts, cut at {debug_right['final_cut_pos']}",
                        (20, right_detail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            right_detail_y += 25
            for i, attempt in enumerate(debug_right['detection_attempts'][:2]):  # Show first 2 attempts with details
                status = "PASS" if attempt['passed'] else "FAIL"
                detail_text = f"  Attempt {i+1}: pos={attempt['position']}, len={attempt['border_length']}, drop={attempt['drop_magnitude']:.4f} - {status}"
                cv2.putText(debug_img, detail_text, (30, right_detail_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 150), 1)
                right_detail_y += 20

                # Show detailed condition breakdown
                proj_max = max(attempt.get('projection_values', [0])) if attempt.get('projection_values') else 0
                conditions = [
                    f"    proj_max: {proj_max:.3f} >= border_thresh: {debug_right.get('border_threshold', 0):.3f}",
                    f"    near_edge: {attempt.get('is_near_edge', False)} (start: {attempt.get('border_start', 0)})",
                    f"    len: {attempt.get('length_ok', False)} grad: {attempt.get('gradient_ok', False)} prom: {attempt.get('prominence_ok', False)}"
                ]
                for j, cond in enumerate(conditions):
                    cv2.putText(debug_img, cond, (30, right_detail_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 100), 1)
                    right_detail_y += 15
                right_detail_y += 10  # Extra spacing between attempts
        else:
            debug_status = f"No attempts (max_cov: {debug_right.get('max_coverage', 0):.3f}, len: {debug_right.get('projection_length', 0)})"
            cv2.putText(debug_img, debug_status, (20, right_detail_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            right_detail_y += 25

    else:
        cv2.putText(debug_img, "Empty debug_right object", (20, right_detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Add projection info at bottom (moved from top to avoid blocking histogram)
    if max_width > 0:
        cv2.putText(debug_img, f"Projection Info: Left len={len(left_proj)}, Right len={len(right_proj)}",
                    (20, debug_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(debug_img, f"Max Coverage: Left={left_max:.3f}, Right={right_max:.3f}",
                    (20, debug_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return debug_img


def _render_combined_debug(roi_bgr: np.ndarray,
                           gray_original: np.ndarray,
                           gray_cleaned: np.ndarray,
                           bin_original: np.ndarray,
                           bin_after: np.ndarray,
                           crop_before_border: np.ndarray,
                           crop_after_border: np.ndarray,
                           xl_proj: int, xr_proj: int, yt_proj: int, yb_proj: int,
                           xl_border: int, xr_border: int, yt_border: int, yb_border: int) -> np.ndarray:
    """Render 1x4 panel: NOISE + CC + PROJ + BORDER rows."""
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return roi_bgr.copy()

    mask_original = (bin_original > 0).astype(np.uint8)
    mask_after = (bin_after > 0).astype(np.uint8)

    panel_h = int(max(120, min(260, round(h * 0.8))))
    scale = panel_h / float(max(1, h))
    base_w = max(1, int(round(w * scale)))

    def resize_panel(img: np.ndarray, interp: int = cv2.INTER_LINEAR, target_w: int | None = None) -> np.ndarray:
        if target_w is None:
            target_w = base_w
        return cv2.resize(img, (target_w, panel_h), interpolation=interp)

    def to_bgr(mask: np.ndarray) -> np.ndarray:
        vis = 255 - (mask.astype(np.uint8) * 255)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def add_border(img: np.ndarray, pad: int = 2) -> np.ndarray:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # === Original 2x4 layout (CC vs Projection) ===
    # CC overlay and masks
    masked_roi = np.full_like(roi_bgr, 255)
    masked_roi[mask_after.astype(bool)] = roi_bgr[mask_after.astype(bool)]
    cc_overlay = masked_roi.copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_original, connectivity=8)
    if num > 1:
        for i in range(1, num):
            x, y, ww, hh, _area = stats[i]
            kept = bool(mask_after[labels == i].any())
            color = (0, 200, 0) if kept else (0, 0, 255)
            cv2.rectangle(cc_overlay, (int(x), int(y)), (int(x + ww - 1), int(y + hh - 1)), color, 1)

    cc_overlay_panel = resize_panel(cc_overlay)
    mask_before_panel = resize_panel(to_bgr(mask_original), interp=cv2.INTER_NEAREST)
    mask_after_panel = resize_panel(to_bgr(mask_after), interp=cv2.INTER_NEAREST)

    # Projection overlay
    projection_overlay = masked_roi.copy()
    cv2.line(projection_overlay, (max(0, int(xl_proj)), 0), (max(0, int(xl_proj)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (max(0, int(xr_proj - 1)), 0), (max(0, int(xr_proj - 1)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yt_proj))), (w - 1, max(0, int(yt_proj))), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yb_proj - 1))), (w - 1, max(0, int(yb_proj - 1))), (0, 0, 255), 1)
    projection_panel = resize_panel(projection_overlay)

    # Projection histograms
    hist_w = base_w
    v_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    vproj = mask_after.sum(axis=0).astype(np.float32)
    if vproj.size:
        vp = vproj / (vproj.max() + 1e-6)
        for col in range(hist_w):
            sx = int(round(col / max(1, hist_w - 1) * max(0, w - 1)))
            bar = int(round(vp[sx] * (panel_h - 1)))
            if bar > 0:
                v_panel[panel_h - bar:panel_h, col] = 0
        xl_bar = int(round(max(0, xl_proj) / max(1, w - 1) * max(0, hist_w - 1)))
        xr_bar = int(round(max(0, xr_proj - 1) / max(1, w - 1) * max(0, hist_w - 1)))
        cv2.line(v_panel, (xl_bar, 0), (xl_bar, panel_h - 1), 128, 1)
        cv2.line(v_panel, (xr_bar, 0), (xr_bar, panel_h - 1), 128, 1)
    vertical_panel = cv2.cvtColor(v_panel, cv2.COLOR_GRAY2BGR)

    h_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    hproj = mask_after.sum(axis=1).astype(np.float32)
    if hproj.size:
        hp = hproj / (hproj.max() + 1e-6)
        for row in range(panel_h):
            sy = int(round(row / max(1, panel_h - 1) * max(0, h - 1)))
            bar = int(round(hp[sy] * (hist_w - 1)))
            if bar > 0:
                h_panel[row, :bar] = 0
        yt_bar = int(round(max(0, yt_proj) / max(1, h - 1) * max(0, panel_h - 1)))
        yb_bar = int(round(max(0, yb_proj - 1) / max(1, h - 1) * max(0, panel_h - 1)))
        cv2.line(h_panel, (0, yt_bar), (hist_w - 1, yt_bar), 128, 1)
        cv2.line(h_panel, (0, yb_bar), (hist_w - 1, yb_bar), 128, 1)
    horizontal_panel = cv2.cvtColor(h_panel, cv2.COLOR_GRAY2BGR)

    # === Border removal row ===
    # Create border removal visualization similar to PROJ
    border_overlay = masked_roi.copy()
    cv2.line(border_overlay, (max(0, int(xl_border)), 0), (max(0, int(xl_border)), h - 1), (255, 0, 0), 1)
    cv2.line(border_overlay, (max(0, int(xr_border - 1)), 0), (max(0, int(xr_border - 1)), h - 1), (255, 0, 0), 1)
    cv2.line(border_overlay, (0, max(0, int(yt_border))), (w - 1, max(0, int(yt_border))), (255, 0, 0), 1)
    cv2.line(border_overlay, (0, max(0, int(yb_border - 1))), (w - 1, max(0, int(yb_border - 1))), (255, 0, 0), 1)
    border_panel = resize_panel(border_overlay)

    # Border horizontal projection analysis with detailed visualization
    border_region = bin_after[yt_proj:yb_proj, xl_proj:xr_proj] if xl_proj < xr_proj and yt_proj < yb_proj else np.zeros((1, 1), dtype=np.uint8)

    # Create comprehensive border projection visualization
    border_h_panel_bgr = _create_border_projection_viz(
        border_region,
        BORDER_REMOVAL_CONFIG,
        max_height=panel_h,
        xl_cut=xl_border - xl_proj,
        xr_cut=xr_border - xl_proj,
        target_width=hist_w
    )

    # Show final crop result with aspect ratio preserved
    if crop_after_border is not None and crop_after_border.size > 0:
        # Calculate aspect ratio preserving scale
        crop_h, crop_w = crop_after_border.shape[:2]
        if crop_h > 0 and crop_w > 0:
            scale_factor = min(panel_h / crop_h, base_w / crop_w)
            new_h = int(crop_h * scale_factor)
            new_w = int(crop_w * scale_factor)
            crop_resized = cv2.resize(crop_after_border, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Center the image in a panel_h x base_w canvas
            crop_after_panel = np.full((panel_h, base_w, 3), 255, dtype=np.uint8)
            y_offset = (panel_h - new_h) // 2
            x_offset = (base_w - new_w) // 2
            crop_after_panel[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
        else:
            crop_after_panel = np.full((panel_h, base_w, 3), 250, dtype=np.uint8)
    else:
        crop_after_panel = np.full((panel_h, base_w, 3), 250, dtype=np.uint8)

    # === Noise removal row ===
    # Show noise cleaning effect by highlighting differences
    noise_diff = cv2.absdiff(gray_original, gray_cleaned)
    noise_overlay = cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR)

    # Highlight removed noise in red
    noise_mask = (noise_diff > 5).astype(np.uint8)  # Threshold for visible differences
    noise_overlay[noise_mask > 0] = [0, 0, 255]  # Red for removed noise

    noise_overlay_panel = resize_panel(noise_overlay)
    gray_original_panel = resize_panel(cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR))
    gray_cleaned_panel = resize_panel(cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR))


    # Labels
    label_w = max(60, hist_w // 4)
    label_color = (245, 245, 245)
    cc_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    proj_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    noise_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    border_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    cv2.putText(cc_label, 'CC', (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(proj_label, 'PROJ', (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(noise_label, 'NOISE', (5, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(border_label, 'BORDER', (2, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2, cv2.LINE_AA)

    def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
        h_, w_ = img.shape[:2]
        if w_ >= target_w:
            return img
        pad_left = (target_w - w_) // 2
        pad_right = target_w - w_ - pad_left
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Build 4 rows: NOISE, CC, PROJ, BORDER
    try:
        # Try to create each border panel component individually for better error reporting
        try:
            border_panel_bordered = add_border(border_panel)
        except Exception as e:
            raise RuntimeError(f"border_panel creation failed: {e}")

        try:
            border_h_panel_bordered = add_border(border_h_panel_bgr)
        except Exception as e:
            raise RuntimeError(f"border_h_panel_bgr creation failed: {e}")

        try:
            crop_after_panel_bordered = add_border(crop_after_panel)
        except Exception as e:
            raise RuntimeError(f"crop_after_panel creation failed: {e}")

        try:
            border_label_bordered = add_border(border_label)
        except Exception as e:
            raise RuntimeError(f"border_label creation failed: {e}")

        border_row = [border_panel_bordered, border_h_panel_bordered, crop_after_panel_bordered, border_label_bordered]

        rows = [
            [add_border(noise_overlay_panel), add_border(gray_original_panel), add_border(gray_cleaned_panel), add_border(noise_label)],
            [add_border(cc_overlay_panel), add_border(mask_before_panel), add_border(mask_after_panel), add_border(cc_label)],
            [add_border(projection_panel), add_border(vertical_panel), add_border(horizontal_panel), add_border(proj_label)],
            border_row
        ]
    except Exception as e:
        # Detailed error reporting for border row creation failure
        print(f"[BORDER ROW ERROR] {type(e).__name__}: {str(e)}")
        # Fallback to 3 rows if border row fails
        rows = [
            [add_border(noise_overlay_panel), add_border(gray_original_panel), add_border(gray_cleaned_panel), add_border(noise_label)],
            [add_border(cc_overlay_panel), add_border(mask_before_panel), add_border(mask_after_panel), add_border(cc_label)],
            [add_border(projection_panel), add_border(vertical_panel), add_border(horizontal_panel), add_border(proj_label)]
        ]

    grid_rows = []
    for row_panels in rows:
        # Pad all panels in this row to same width
        max_w = max(panel.shape[1] for panel in row_panels)
        padded_panels = [pad_to_width(panel, max_w) for panel in row_panels]

        # Combine with gaps
        gap = 6
        gap_col = np.full((padded_panels[0].shape[0], gap, 3), 255, dtype=np.uint8)

        row_img = padded_panels[0]
        for panel in padded_panels[1:]:
            row_img = np.hstack([row_img, gap_col, panel])
        grid_rows.append(row_img)

    try:
        result = np.vstack(grid_rows)
        return result
    except Exception as e:
        # Detailed error reporting for vstack failure
        print(f"[VSTACK ERROR] Failed to combine {len(grid_rows)} rows: {type(e).__name__}: {str(e)}")
        if len(grid_rows) >= 4:
            print(f"[VSTACK ERROR] Row shapes: {[row.shape for row in grid_rows]}")
        # Return first 3 rows if vstack fails
        try:
            return np.vstack(grid_rows[:3])
        except Exception as e2:
            print(f"[VSTACK ERROR] Even 3-row fallback failed: {type(e2).__name__}: {str(e2)}")
            # Last resort: return a blank image
            return np.full((400, 800, 3), 255, dtype=np.uint8)

def run_on_image(image_path: str, output_dir: str, expected_text: str | None = None,
                 framework: str = 'livetext', recognition_level: str = 'accurate',
                 language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    if _ocrmac is None:
        return {'success': False, 'error': f'ocrmac 未安装: {_OCRMAC_IMPORT_ERROR}'}
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': f'cannot read: {image_path}'}
    os.makedirs(output_dir, exist_ok=True)
    langs = language_preference or ['zh-Hans']
    try:
        o = _ocrmac.OCR(image_path, framework='livetext', recognition_level=recognition_level,
                        language_preference=langs, detail=True)
        res = o.recognize()
    except Exception as e:
        return {'success': False, 'error': f'LiveText 调用失败: {e}'}
    if not res:
        return {'success': False, 'error': 'LiveText 无检测结果'}
    W, H = int(o.image.width), int(o.image.height)
    boxes = []
    for i, tup in enumerate(res):
        if not isinstance(tup, (list, tuple)) or len(tup) != 3:
            continue
        text, conf, nb = tup
        x = int(round(nb[0] * W))
        y_top = int(round((1.0 - nb[1] - nb[3]) * H))
        w = int(round(nb[2] * W))
        h = int(round(nb[3] * H))
        x = max(0, min(W - 1, x))
        y_top = max(0, min(H - 1, y_top))
        w = max(1, min(W - x, w))
        h = max(1, min(H - y_top, h))
        boxes.append({
            'order': i+1,
            'text': str(text),
            'confidence': float(conf),
            'x': x,
            'y': y_top,
            'w': w,
            'h': h,
            'normalized_bbox': [float(nb[0]), float(nb[1]), float(nb[2]), float(nb[3])]
        })
    boxes.sort(key=lambda b: (b['y'], b['x']))

    chars_meta: List[Dict[str, Any]] = []
    refined_boxes: List[Tuple[int,int,int,int]] = []
    dbg_enabled = bool(SEGMENT_REFINE_CONFIG.get('debug_visualize', False))
    dbg_dirname = str(SEGMENT_REFINE_CONFIG.get('debug_dirname', 'debug'))

    raw_mode = str(SEGMENT_REFINE_CONFIG.get('mode', 'ccprojection')).lower()
    mode_alias = {
        'ccprojection': 'ccprojection',
        'cc_projection': 'ccprojection',
        'projection': 'ccprojection',
        'projection_only': 'projection_only',
        'cc_debug': 'cc_debug',
    }
    mode = mode_alias.get(raw_mode, raw_mode)
    if mode not in {'ccprojection', 'projection_only', 'cc_debug'}:
        mode = 'ccprojection'
    expand_cfg = SEGMENT_REFINE_CONFIG.get('expand_px', 0)
    if isinstance(expand_cfg, dict):
        ex_left = int(max(0, expand_cfg.get('left', 0)))
        ex_right = int(max(0, expand_cfg.get('right', 0)))
        ex_top = int(max(0, expand_cfg.get('top', 0)))
        ex_bottom = int(max(0, expand_cfg.get('bottom', 0)))
    else:
        ex_left = ex_right = ex_top = ex_bottom = int(max(0, expand_cfg))

    for b in boxes:
        x0 = max(0, b['x'] - ex_left); y0 = max(0, b['y'] - ex_top)
        x1 = min(W, b['x'] + b['w'] + ex_right); y1 = min(H, b['y'] + b['h'] + ex_bottom)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Noise patch removal (before binarization)
        gray_cleaned = gray.copy()
        if NOISE_REMOVAL_CONFIG.get('enabled', True):
            gray_cleaned = remove_noise_patches(gray_cleaned, NOISE_REMOVAL_CONFIG)

        bin_before = binarize(
            gray_cleaned,
            mode=str(PROJECTION_TRIM_CONFIG.get('binarize', 'otsu')).lower(),
            adaptive_block=int(PROJECTION_TRIM_CONFIG.get('adaptive_block', 31)),
            adaptive_C=int(PROJECTION_TRIM_CONFIG.get('adaptive_C', 3)),
        )

        # choose CC filtering according to mode
        if mode == 'projection_only':
            bin_after = bin_before.copy()  # skip CC filtering
        else:
            bin_after = bin_before.copy()
            refine_binary_components(bin_after, CC_FILTER_CONFIG)
        if mode == 'cc_debug':
            # no projection trimming; use full ROI as crop
            xl, xr, yt, yb = 0, roi.shape[1], 0, roi.shape[0]
        else:
            xl, xr, yt, yb = trim_projection_from_bin(bin_after, PROJECTION_TRIM_CONFIG)

        # Store projection coordinates for debug
        xl_proj, xr_proj, yt_proj, yb_proj = xl, xr, yt, yb

        # Border removal (after projection trimming)
        xl_border, xr_border, yt_border, yb_border = xl, xr, yt, yb
        if BORDER_REMOVAL_CONFIG.get('enabled', True):
            # Apply border trimming to the projection-trimmed region
            border_bin = bin_after[yt:yb, xl:xr]
            xl_b, xr_b, yt_b, yb_b = trim_border_from_bin(border_bin, BORDER_REMOVAL_CONFIG)
            # Adjust coordinates back to full ROI space
            xl_border = xl + xl_b
            xr_border = xl + xr_b
            yt_border = yt + yt_b
            yb_border = yt + yb_b

        fp = int(SEGMENT_REFINE_CONFIG.get('final_pad', 0))
        xl_final = max(0, xl_border - fp)
        xr_final = min(roi.shape[1], xr_border + fp)
        yt_final = max(0, yt_border - fp)
        yb_final = min(roi.shape[0], yb_border + fp)
        rx, ry, rw, rh = x0 + int(xl_final), y0 + int(yt_final), max(1, int(xr_final - xl_final)), max(1, int(yb_final - yt_final))
        refined_boxes.append((rx, ry, rw, rh))

        # Create final processed image: apply NOISE removal + CC filtering effects
        # Start with noise-cleaned grayscale image
        processed_gray = gray_cleaned.copy()

        # Apply CC filtering effects by masking with bin_after
        # Where bin_after is 0 (removed by CC), set to white background
        cc_mask = (bin_after > 0)
        processed_gray[~cc_mask] = 255

        # Convert to BGR for final output
        processed_roi = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

        # Create crops from processed image
        crop_before_border = processed_roi[yt_proj:yb_proj, xl_proj:xr_proj]
        crop_after_border = processed_roi[yt_final:yb_final, xl_final:xr_final]
        crop = crop_after_border

        fname = f"char_{b['order']:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), crop)
        if dbg_enabled:
            try:
                # Pass all stages to show the complete pipeline (4 stages: noise, cc, proj, border)
                dbg_img = _render_combined_debug(roi, gray, gray_cleaned, bin_before, bin_after,
                                                crop_before_border, crop_after_border,
                                                int(xl_proj), int(xr_proj), int(yt_proj), int(yb_proj),
                                                int(xl_border), int(xr_border), int(yt_border), int(yb_border))
                dbg_name = f"{os.path.splitext(fname)[0]}_debug.png"
                out_dbg = os.path.join(output_dir, dbg_dirname)
                _ensure_dir(out_dbg)
                cv2.imwrite(os.path.join(out_dbg, dbg_name), dbg_img)

                # Generate border detection debug image (verbose version)
                if BORDER_REMOVAL_CONFIG.get('debug_verbose', False):
                    border_debug_img = _create_border_debug_image(
                        bin_after[yt_proj:yb_proj, xl_proj:xr_proj],
                        int(xl_border), int(xr_border),
                        int(xl_proj), int(xr_proj),
                        BORDER_REMOVAL_CONFIG
                    )
                    border_dbg_name = f"{os.path.splitext(fname)[0]}_border_debug.png"
                    cv2.imwrite(os.path.join(out_dbg, border_dbg_name), border_debug_img)
            except Exception as e:
                # Log debug failure with character information
                error_msg = f"DEBUG FAILURE for {fname}: {type(e).__name__}: {str(e)}"
                print(f"[DEBUG ERROR] {error_msg}")
                # Also write error to a log file
                try:
                    error_log_path = os.path.join(output_dir, dbg_dirname, "debug_errors.log")
                    _ensure_dir(os.path.join(output_dir, dbg_dirname))
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{error_msg}\n")
                except:
                    pass
        chars_meta.append({
            'filename': fname,
            'bbox': (b['x'], b['y'], b['w'], b['h']),
            'refined_bbox': (rx, ry, rw, rh),
            'text_hint': b['text'],
            'confidence': b['confidence'],
            'normalized_bbox': b.get('normalized_bbox')
        })

    # Create comparison overlay: original clean image vs assembled processed characters
    overlay_original = img.copy()  # Clean original image without any annotations

    # Create processed image assembly: white background with processed characters placed back
    overlay_processed = np.full_like(img, 255, dtype=np.uint8)  # Start with white background

    # Place processed characters back in their original positions
    if chars_meta:
        for idx, meta in enumerate(chars_meta):
            crop_path = os.path.join(output_dir, meta['filename'])
            if os.path.exists(crop_path):
                crop_img = cv2.imread(crop_path)
                if crop_img is not None:
                    # Get the refined bbox position where the character should be placed
                    rx, ry, rw, rh = meta['refined_bbox']

                    # Resize crop to fit the refined bbox if necessary
                    if crop_img.shape[:2] != (rh, rw):
                        crop_img_resized = cv2.resize(crop_img, (rw, rh))
                    else:
                        crop_img_resized = crop_img

                    # Ensure the placement area is within image bounds
                    y_start = max(0, ry)
                    y_end = min(overlay_processed.shape[0], ry + rh)
                    x_start = max(0, rx)
                    x_end = min(overlay_processed.shape[1], rx + rw)

                    # Adjust crop size if placement area is clipped
                    crop_h = y_end - y_start
                    crop_w = x_end - x_start
                    if crop_h > 0 and crop_w > 0:
                        if crop_h != rh or crop_w != rw:
                            crop_img_resized = cv2.resize(crop_img_resized, (crop_w, crop_h))

                        # Place the processed character in the refined position
                        overlay_processed[y_start:y_end, x_start:x_end] = crop_img_resized

    # Create side-by-side comparison
    gap_width = 20
    gap = np.full((overlay_original.shape[0], gap_width, 3), 255, dtype=np.uint8)
    comparison = np.hstack([overlay_original, gap, overlay_processed])

    overlay_path = os.path.join(output_dir, 'overlay.png')
    cv2.imwrite(overlay_path, comparison)

    stats = {
        'framework': 'livetext',
        'recognition_level': recognition_level,
        'char_candidates': len(boxes),
        'expected_text_len': len(expected_text) if expected_text else None,
        'image_size': [W, H],
        'refine_config': {
            'mode': mode,
            'expand_px': {
                'left': ex_left,
                'right': ex_right,
                'top': ex_top,
                'bottom': ex_bottom,
            },
            'final_pad': int(SEGMENT_REFINE_CONFIG.get('final_pad', 0)),
        },
        'projection_trim': PROJECTION_TRIM_CONFIG,
    }
    return {
        'success': True,
        'character_count': len(chars_meta),
        'characters': chars_meta,
        'stats': stats,
        'overlay': overlay_path,
    }


def run_on_ocr_regions(dataset: str | None = None,
                       expected_texts: Dict[str, str] | None = None,
                       framework: str = 'livetext', recognition_level: str = 'accurate',
                       language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    def _find_region_images(base_ocr_dir: str, dataset: str | None = None) -> List[Tuple[str, str, str]]:
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
        results.sort(key=lambda t: (t[0], t[1]))
        return results

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
            res = run_on_image(img_path, out_dir, expected_text=exp_text, framework=framework,
                               recognition_level=recognition_level, language_preference=language_preference)
            with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            processed.append({'dataset': ds, 'region': region_name, 'out_dir': out_dir,
                              'count': res.get('character_count', 0)})
        except Exception as e:
            errors.append({'dataset': ds, 'region': region_name, 'error': str(e)})
    return {'processed': processed, 'errors': errors}
