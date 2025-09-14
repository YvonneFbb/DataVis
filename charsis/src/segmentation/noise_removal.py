"""
Noise removal for ancient text images.

Remove light-colored impurities and stains that interfere with text recognition
while preserving legitimate character strokes.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import cv2


def remove_noise_patches(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Remove light-colored noise patches from grayscale image.

    Simple strategy for ancient texts:
    1. Find medium-gray pixels (potential noise)
    2. Check if they connect to dark strokes (text edges vs isolated noise)
    3. Remove isolated light patches that don't support text

    Args:
        gray_img: Grayscale image (uint8)
        params: Configuration parameters

    Returns:
        Cleaned grayscale image
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img

    return _connectivity_based_noise_removal(gray_img, params)


def _dual_threshold_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """Dual-threshold based noise removal."""
    h, w = gray_img.shape[:2]
    if h == 0 or w == 0:
        return gray_img

    # Parameters
    strict_ratio = float(params.get('strict_threshold_ratio', 0.7))
    loose_ratio = float(params.get('loose_threshold_ratio', 1.2))
    min_area = int(params.get('min_noise_area', 5))
    max_area_ratio = float(params.get('max_noise_area_ratio', 0.1))
    shape_verify = bool(params.get('shape_verification', True))

    # Calculate Otsu threshold
    _, otsu_bin = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]

    # Dual thresholds
    strict_thresh = int(otsu_thresh * strict_ratio)
    loose_thresh = int(otsu_thresh * loose_ratio)

    # Create masks
    strict_mask = (gray_img <= strict_thresh).astype(np.uint8)
    loose_mask = (gray_img <= loose_thresh).astype(np.uint8)

    # Noise candidates: appear in loose but not in strict
    noise_candidates = loose_mask & ~strict_mask

    if noise_candidates.sum() == 0:
        return gray_img

    # Connected component analysis on noise candidates
    num, labels, stats, _ = cv2.connectedComponentsWithStats(noise_candidates, connectivity=8)

    result = gray_img.copy()
    max_area = int(h * w * max_area_ratio)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Size filtering
        if area < min_area or area > max_area:
            continue

        # Shape verification
        if shape_verify:
            if not _is_likely_noise(gray_img, labels == i, x, y, ww, hh, area, params):
                continue

        # Mark as noise - set to local background color
        component_mask = (labels == i)
        background_color = _estimate_local_background(gray_img, component_mask)
        result[component_mask] = background_color

    return result


def _statistical_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """Statistical analysis based noise removal."""
    # Calculate histogram to find main text range
    hist, bins = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))

    # Find dominant dark range (main text)
    dark_peak = np.argmax(hist[:128])  # Peak in dark range
    text_upper_bound = min(dark_peak + 50, 180)  # Conservative upper bound

    # Noise range parameters
    noise_range = params.get('noise_gray_range', [150, 220])
    min_area = int(params.get('min_noise_area', 5))

    # Create noise mask
    noise_mask = ((gray_img >= noise_range[0]) & (gray_img <= noise_range[1])).astype(np.uint8)

    # Connected component filtering
    num, labels, stats, _ = cv2.connectedComponentsWithStats(noise_mask, connectivity=8)

    result = gray_img.copy()
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area >= min_area:
            component_mask = (labels == i)
            background_color = _estimate_local_background(gray_img, component_mask)
            result[component_mask] = background_color

    return result


def _hybrid_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """Combine dual-threshold and statistical methods."""
    # Apply dual-threshold first
    result1 = _dual_threshold_noise_removal(gray_img, params)

    # Then apply statistical filtering
    result2 = _statistical_noise_removal(result1, params)

    return result2


def _is_likely_noise(gray_img: np.ndarray, component_mask: np.ndarray,
                    x: int, y: int, w: int, h: int, area: int, params: Dict) -> bool:
    """Analyze component characteristics to determine if it's likely noise."""

    # Geometric features
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    perimeter = cv2.countNonZero(cv2.Canny(component_mask.astype(np.uint8) * 255, 50, 150))
    compactness = 4 * np.pi * area / max(perimeter * perimeter, 1) if perimeter > 0 else 0

    # Shape-based filtering
    min_compactness = float(params.get('min_compactness', 0.1))

    # Very elongated or very scattered regions are likely noise
    if aspect_ratio > 8.0:  # Too elongated
        return True

    if compactness < min_compactness:  # Too scattered
        return True

    # Analyze gray level distribution within component
    component_pixels = gray_img[component_mask]
    if len(component_pixels) > 0:
        mean_gray = np.mean(component_pixels)
        std_gray = np.std(component_pixels)

        # Noise tends to have higher mean gray level and lower variation
        if mean_gray > 180 and std_gray < 20:  # Light and uniform
            return True

    return False


def _estimate_local_background(gray_img: np.ndarray, component_mask: np.ndarray,
                              radius: int = 5) -> int:
    """Estimate local background color around a component."""
    # Dilate the mask to get surrounding area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    dilated = cv2.dilate(component_mask.astype(np.uint8), kernel)

    # Background is dilated area minus original component
    background_mask = dilated & ~component_mask.astype(np.uint8)

    if background_mask.sum() > 0:
        background_pixels = gray_img[background_mask.astype(bool)]
        # Use median for robustness
        return int(np.median(background_pixels))
    else:
        # Fallback: use surrounding region mean
        y_coords, x_coords = np.where(component_mask)
        if len(y_coords) > 0:
            y_min, y_max = max(0, y_coords.min()-radius), min(gray_img.shape[0], y_coords.max()+radius+1)
            x_min, x_max = max(0, x_coords.min()-radius), min(gray_img.shape[1], x_coords.max()+radius+1)

            surrounding = gray_img[y_min:y_max, x_min:x_max]
            return int(np.median(surrounding))

    return 240  # Default light background


def _connectivity_based_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Distance and structure-based noise removal for ancient texts.

    Advanced strategy:
    - Text edges are close to and structurally support dark strokes
    - Noise patches are distant from or structurally disconnected from main strokes
    - Use distance transform and structural analysis
    """
    h, w = gray_img.shape[:2]
    if h == 0 or w == 0:
        return gray_img

    # Parameters
    dark_threshold = int(params.get('dark_stroke_threshold', 100))    # Deep strokes
    light_threshold = int(params.get('light_noise_threshold', 180))   # Light noise candidates
    min_noise_area = int(params.get('min_noise_area', 5))
    max_noise_area = int(params.get('max_noise_area', 200))

    # Create masks
    dark_strokes = (gray_img <= dark_threshold).astype(np.uint8)    # Deep text strokes
    light_candidates = ((gray_img > dark_threshold) &
                       (gray_img < light_threshold)).astype(np.uint8)  # Medium gray candidates

    if light_candidates.sum() == 0:
        return gray_img

    # Find connected components in light candidates
    num, labels, stats, _ = cv2.connectedComponentsWithStats(light_candidates, connectivity=8)

    result = gray_img.copy()

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Size filtering
        if area < min_noise_area or area > max_noise_area:
            continue

        # Get this component's mask
        component_mask = (labels == i)

            # Advanced analysis: distance + structural relationship
        is_noise = _analyze_component_relationship(
            component_mask, dark_strokes, gray_img, params
        )

        if is_noise:
            # This is likely noise - remove it
            background_color = _estimate_local_background(gray_img, component_mask)
            result[component_mask] = background_color

    return result


def _analyze_component_relationship(component_mask: np.ndarray, dark_strokes: np.ndarray,
                                   gray_img: np.ndarray, params: Dict) -> bool:
    """
    简化的噪声判断：主要基于面积大小

    策略：汉字周围大块的色块就是杂质，小的区域可以认为是边缘

    返回 True 表示是噪声，False 表示可能是文字边缘
    """

    area = np.sum(component_mask)
    if area == 0:
        return True

    # 主要判断：面积过大的认为是杂质
    large_noise_threshold = int(params.get('large_noise_area_threshold', 50))
    if area > large_noise_threshold:
        return True

    # 辅助判断：极端形状的小块也可能是噪声
    y_coords, x_coords = np.where(component_mask)
    if len(y_coords) > 0:
        h_span = y_coords.max() - y_coords.min() + 1
        w_span = x_coords.max() - x_coords.min() + 1

        # 极端细长的小块可能是噪声
        aspect_ratio = max(h_span, w_span) / max(min(h_span, w_span), 1)
        if aspect_ratio > 15.0 and area < 20:  # 细长且很小
            return True

    return False  # 小面积且形状合理，保留为文字边缘