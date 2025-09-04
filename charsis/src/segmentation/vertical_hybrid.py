"""
Robust vertical single-character segmentation for OCR region images.

Designed and tuned for data/results/ocr/demo/region_images/region_001.jpg.
"""
from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 10)
    return bin_img


def _crop_horizontal_foreground(binary: np.ndarray, margin: int = 2) -> Tuple[np.ndarray, Tuple[int, int]]:
    vproj = np.sum(binary, axis=0)
    cols = np.where(vproj > 0)[0]
    if len(cols) == 0:
        return binary, (0, binary.shape[1])
    x1 = max(0, cols[0] - margin)
    x2 = min(binary.shape[1], cols[-1] + margin + 1)
    return binary[:, x1:x2], (x1, x2)


def _deskew_small_angles(gray: np.ndarray, binary: np.ndarray, angle_range=(-3.0, 3.0), step=0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = gray.shape
    best_score = -1
    best_angle = 0.0
    best_gray = gray
    best_bin = binary

    def score_of(bin_img: np.ndarray) -> float:
        proj = np.sum(bin_img, axis=1) // 255
        sm = gaussian_filter1d(proj.astype(float), sigma=max(1.0, h / 200))
        inv = np.max(sm) - sm
        peaks, props = find_peaks(inv, distance=max(3, int(h / 50)), prominence=np.max(inv) * 0.05)
        if len(peaks) == 0:
            return 0.0
        prom = np.mean(props.get('prominences', np.array([0.0])))
        return len(peaks) * prom

    for ang in np.arange(angle_range[0], angle_range[1] + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        g_r = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b_r = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sc = score_of(b_r)
        if sc > best_score:
            best_score, best_angle, best_gray, best_bin = sc, ang, g_r, b_r
    return best_gray, best_bin, best_angle


def _horizontal_gaps(bin_img: np.ndarray, mean_char_h_hint: float | None = None) -> List[int]:
    proj = np.sum(bin_img, axis=1) // 255
    h = len(proj)
    sigma = max(1.5, (mean_char_h_hint or (h / 20)) / 6)
    sm = gaussian_filter1d(proj.astype(float), sigma=sigma)
    inv = np.max(sm) - sm
    min_dist = max(3, int((mean_char_h_hint or (h / 20)) / 2))
    peaks, _ = find_peaks(inv, distance=min_dist, prominence=np.max(inv) * 0.05)
    return peaks.tolist()


def _segments_from_gaps(h: int, gaps: List[int], min_seg_h: int) -> List[Tuple[int, int]]:
    if not gaps:
        return [(0, h)]
    segs: List[Tuple[int, int]] = []
    prev = 0
    for g in gaps:
        if g - prev >= min_seg_h:
            segs.append((prev, g))
            prev = g
    if h - prev >= min_seg_h:
        segs.append((prev, h))
    return segs


def _watershed_split(bin_img: np.ndarray) -> List[Tuple[int, int]]:
    h, w = bin_img.shape
    dist = cv2.distanceTransform(bin_img, distanceType=cv2.DIST_L2, maskSize=5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(bin_img, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    segs: List[Tuple[int, int]] = []
    for m in range(2, num_markers + 1):
        ys, xs = np.where(markers == m)
        if len(ys) == 0:
            continue
        y1, y2 = int(np.min(ys)), int(np.max(ys) + 1)
        if y2 - y1 >= 4:
            segs.append((y1, y2))
    segs = sorted(segs, key=lambda t: t[0])
    merged: List[Tuple[int, int]] = []
    for s in segs:
        if not merged or s[0] > merged[-1][1] + 1:
            merged.append(s)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
    return merged


def segment_vertical_single_column(image: np.ndarray,
                                   expected_chars: int | None = None,
                                   debug: bool = True) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    h, w = image.shape[:2]
    gray = _to_gray(image)
    gray = _clahe(gray)
    bin0 = _adaptive_binarize(gray)
    bin1, (x1, x2) = _crop_horizontal_foreground(bin0)
    gray_cropped = gray[:, x1:x2]

    gray2, bin2, angle = _deskew_small_angles(gray_cropped, bin1)

    mean_hint = max(12, int(h / 20))
    gaps = _horizontal_gaps(bin2, mean_char_h_hint=mean_hint)
    min_seg_h = max(6, int((h / 30)))
    segs = _segments_from_gaps(bin2.shape[0], gaps, min_seg_h=min_seg_h)

    if expected_chars is not None and expected_chars > 0 and len(segs) < max(2, expected_chars // 2):
        ws = _watershed_split(bin2)
        if len(ws) > len(segs):
            segs = ws

    boxes: List[Tuple[int, int, int, int]] = []
    for (y1, y2) in sorted(segs, key=lambda t: t[0]):
        y1c = max(0, min(y1, h - 1))
        y2c = max(y1c + 1, min(y2, h))
        boxes.append((x1, y1c, x2 - x1, y2c - y1c))

    stats = {
        'angle': angle,
        'segments': len(boxes),
        'x_crop': (x1, x2)
    }

    return boxes, stats


def run_on_image(image_path: str, output_dir: str, expected_text: str | None = None) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': f'cannot read: {image_path}'}
    expected = len(expected_text) if expected_text else None
    boxes, stats = segment_vertical_single_column(img, expected_chars=expected, debug=True)

    os.makedirs(output_dir, exist_ok=True)
    chars = []
    for i, (x, y, w, h) in enumerate(boxes):
        char_img = img[y:y + h, x:x + w]
        name = f"char_{i + 1:04d}.png"
        cv2.imwrite(os.path.join(output_dir, name), char_img)
        chars.append({'filename': name, 'bbox': (int(x), int(y), int(w), int(h))})

    # debug overlay
    overlay = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, 'overlay.png'), overlay)

    return {
        'success': True,
        'character_count': len(chars),
        'characters': chars,
        'stats': stats,
        'overlay': os.path.join(output_dir, 'overlay.png')
    }


if __name__ == '__main__':
    # Default demo path based on repo layout
    demo_path = 'data/results/ocr/demo/region_images/region_001.jpg'
    out_dir = 'data/results/segments/demo/region_001'
    os.makedirs(out_dir, exist_ok=True)
    res = run_on_image(demo_path, out_dir)
    print(res)
