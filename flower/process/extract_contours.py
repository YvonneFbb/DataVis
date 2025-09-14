#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract object contours from process/demo.png and export:
- Binary mask (PNG)
- Contour overlay over original image (PNG)
- Optional SVG polygons

Usage (macOS / zsh):
  python process/extract_contours.py \
    --input process/demo.png \
    --method auto  \
    --invert auto \
    --min-area 0.005 \
    --save-svg

Notes:
- --method: auto | otsu | adaptive | canny
- --invert: auto | true | false
- --min-area: float as ratio to image area (e.g., 0.005 = 0.5%)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def auto_invert(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Decide whether foreground is bright or dark by sampling
    mean_fg = np.mean(gray[mask > 0]) if np.any(mask > 0) else 255
    mean_bg = np.mean(gray[mask == 0]) if np.any(mask == 0) else 0
    if mean_fg < mean_bg:
        return 255 - mask
    return mask


def threshold_otsu(gray: np.ndarray, invert: str) -> np.ndarray:
    _t, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if invert == 'true':
        bin_img = 255 - bin_img
    elif invert == 'auto':
        bin_img = auto_invert(gray, bin_img)
    return bin_img


def threshold_adaptive(gray: np.ndarray, invert: str) -> np.ndarray:
    bin_img = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        21, 5
    )
    if invert == 'true':
        bin_img = 255 - bin_img
    elif invert == 'auto':
        bin_img = auto_invert(gray, bin_img)
    return bin_img


def edges_canny(gray: np.ndarray) -> np.ndarray:
    v = float(np.median(gray))
    lower, upper = max(0, 0.66 * v), min(255, 1.33 * v)
    edges = cv.Canny(gray, lower, upper)
    return edges


def morph_cleanup(bin_img: np.ndarray, close_k=7, open_k=5, close_iter=2, open_iter=1) -> np.ndarray:
    close_kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k))
    open_kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_k, open_k))
    img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, close_kern, iterations=close_iter)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, open_kern, iterations=open_iter)
    return img


def biggest_component_mask(bin_img: np.ndarray) -> np.ndarray:
    # Keep the largest connected component (useful when background noise exists)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels <= 1:
        return bin_img
    # Skip background label 0
    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return mask


def contours_from_mask(bin_img: np.ndarray, min_area_px: float, mode=cv.RETR_EXTERNAL, chain=cv.CHAIN_APPROX_NONE):
    contours, _ = cv.findContours(bin_img, mode, chain)
    filtered = [c for c in contours if cv.contourArea(c) >= min_area_px]
    filtered.sort(key=cv.contourArea, reverse=True)
    return filtered


def simplify_contour(c: np.ndarray, eps_ratio: float = 0.01) -> np.ndarray:
    peri = cv.arcLength(c, True)
    eps = eps_ratio * peri
    approx = cv.approxPolyDP(c, eps, True)
    return approx


def draw_overlay(image: np.ndarray, contours: list[np.ndarray], color=(0, 255, 0), thickness=2) -> np.ndarray:
    overlay = image.copy()
    # Use anti-aliased lines where supported by drawing polylines
    for c in contours:
        pts = c.reshape(-1, 1, 2)
        cv.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv.LINE_AA)
    return overlay


# -------- Image enhancement (brightness/color/contrast) ---------
def apply_clahe_y(image: np.ndarray, clip_limit: float = 4.0, tile_grid: int = 8) -> np.ndarray:
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    y_clahe = clahe.apply(y)
    out = cv.merge((y_clahe, cr, cb))
    return cv.cvtColor(out, cv.COLOR_YCrCb2BGR)


def adjust_saturation(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv.split(hsv)
    s *= float(factor)
    s = np.clip(s, 0, 255)
    hsv_enhanced = cv.merge((h, s, v)).astype(np.uint8)
    return cv.cvtColor(hsv_enhanced, cv.COLOR_HSV2BGR)


def apply_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-3:
        return image
    inv = 1.0 / float(gamma)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype('uint8')
    return cv.LUT(image, table)


def enhance_image(image: np.ndarray, enable=True, clahe_clip=4.0, clahe_tile=8, sat_factor=1.2, gamma=1.0) -> np.ndarray:
    if not enable:
        return image
    out = apply_clahe_y(image, clip_limit=clahe_clip, tile_grid=clahe_tile)
    out = adjust_saturation(out, factor=sat_factor)
    out = apply_gamma(out, gamma=gamma)
    return out


# -------- Texture edges extraction ---------
def edges_sobel(gray: np.ndarray, ksize: int = 3, thresh: str | float = 'auto') -> np.ndarray:
    sx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    sy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    mag = cv.magnitude(sx, sy)
    mag_u8 = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if thresh == 'auto':
        _, edges = cv.threshold(mag_u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        _, edges = cv.threshold(mag_u8, float(thresh), 255, cv.THRESH_BINARY)
    return edges


def edges_laplacian(gray: np.ndarray, ksize: int = 3, thresh: str | float = 'auto') -> np.ndarray:
    lap = cv.Laplacian(gray, cv.CV_32F, ksize=ksize)
    mag = np.abs(lap)
    mag_u8 = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if thresh == 'auto':
        _, edges = cv.threshold(mag_u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        _, edges = cv.threshold(mag_u8, float(thresh), 255, cv.THRESH_BINARY)
    return edges


def edges_canny_auto(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    lower = max(0, (1.0 - sigma) * v)
    upper = min(255, (1.0 + sigma) * v)
    return cv.Canny(gray, lower, upper)


def extract_texture_edges(image: np.ndarray,
                          method: str = 'canny',
                          sigma: float = 0.33,
                          sobel_ksize: int = 3,
                          lap_ksize: int = 3,
                          thresh: str | float = 'auto',
                          close_k: int = 3,
                          blur_ksize: int = 3) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == 'canny':
        edges = edges_canny_auto(gray, sigma=sigma)
    elif method == 'sobel':
        edges = edges_sobel(gray, ksize=sobel_ksize, thresh=thresh)
    elif method == 'laplacian':
        edges = edges_laplacian(gray, ksize=lap_ksize, thresh=thresh)
    elif method == 'fusion':
        e1 = edges_canny_auto(gray, sigma=sigma)
        e2 = edges_sobel(gray, ksize=sobel_ksize, thresh=thresh)
        e3 = edges_laplacian(gray, ksize=lap_ksize, thresh=thresh)
        edges = cv.bitwise_or(e1, cv.bitwise_or(e2, e3))
    else:
        edges = edges_canny_auto(gray, sigma=sigma)

    if close_k and close_k > 1:
        kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k))
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kern, iterations=1)

    # Ensure binary 0/255
    edges = (edges > 0).astype(np.uint8) * 255
    return edges


def save_svg(contours: list[np.ndarray], svg_path: Path, width: int, height: int):
    def contour_to_path(c: np.ndarray) -> str:
        pts = c.reshape(-1, 2)
        if len(pts) == 0:
            return ''
        d = [f"M {pts[0,0]} {pts[0,1]}"]
        for x, y in pts[1:]:
            d.append(f"L {x} {y}")
        d.append("Z")
        return ' '.join(d)

    paths = [contour_to_path(c) for c in contours if c.size > 0]
    content = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<g fill='none' stroke='red' stroke-width='2'>",
        *[f"  <path d='{d}' />" for d in paths if d],
        "</g>",
        "</svg>"
    ]
    svg_path.write_text('\n'.join(content), encoding='utf-8')


def run(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    img = cv.imread(str(input_path))
    if img is None:
        print("Failed to read image.")
        sys.exit(1)

    h, w = img.shape[:2]

    # Enhance image if requested
    img_enh = enhance_image(
        img,
        enable=args.enhance,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        sat_factor=args.sat_factor,
        gamma=args.gamma,
    )

    gray = cv.cvtColor(img_enh if args.use_enhanced_for_mask else img, cv.COLOR_BGR2GRAY)
    # Mild denoise while keeping edges
    if args.bilateral:
        gray = cv.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    else:
        gray = cv.GaussianBlur(gray, (5, 5), 0)

    method = args.method
    if method == 'auto':
        # quick heuristic: try Otsu; if too sparse/dense, fallback to adaptive or canny
        bin_img = threshold_otsu(gray, args.invert)
        fill_ratio = float(np.mean(bin_img > 0))
        if fill_ratio < 0.02 or fill_ratio > 0.98:
            # bad separation, try adaptive
            bin_img = threshold_adaptive(gray, args.invert)
            fill_ratio = float(np.mean(bin_img > 0))
            if fill_ratio < 0.02 or fill_ratio > 0.98:
                # fallback to edges then close
                edges = edges_canny(gray)
                bin_img = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=2)
    elif method == 'otsu':
        bin_img = threshold_otsu(gray, args.invert)
    elif method == 'adaptive':
        bin_img = threshold_adaptive(gray, args.invert)
    elif method == 'canny':
        edges = edges_canny(gray)
        bin_img = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=2)
    else:
        print(f"Unknown method: {method}")
        sys.exit(2)

    if args.morph:
        bin_img = morph_cleanup(bin_img, close_k=args.close_k, open_k=args.open_k,
                                close_iter=args.close_iter, open_iter=args.open_iter)

    if args.keep_biggest:
        bin_img = biggest_component_mask(bin_img)

    # Ensure binary
    bin_img = (bin_img > 0).astype(np.uint8) * 255

    min_area_px = args.min_area * (w * h)
    chain_map = {
        'none': cv.CHAIN_APPROX_NONE,
        'simple': cv.CHAIN_APPROX_SIMPLE,
        'tc89': cv.CHAIN_APPROX_TC89_KCOS,
        'tc89_l1': cv.CHAIN_APPROX_TC89_L1,
    }
    chain_flag = chain_map.get(args.chain, cv.CHAIN_APPROX_NONE)

    contours = contours_from_mask(
        bin_img,
        min_area_px=min_area_px,
        mode=cv.RETR_EXTERNAL if args.external else cv.RETR_TREE,
        chain=chain_flag,
    )

    if args.simplify > 0:
        contours = [simplify_contour(c, eps_ratio=args.simplify) for c in contours]

    overlay = draw_overlay(img, contours, color=(0, 255, 0), thickness=args.thickness)

    # Texture edges extraction on enhanced image
    if args.texture:
        tex_edges = extract_texture_edges(
            img_enh if args.use_enhanced_for_texture else img,
            method=args.texture_method,
            sigma=args.texture_sigma,
            sobel_ksize=args.texture_sobel_ksize,
            lap_ksize=args.texture_lap_ksize,
            thresh=args.texture_thresh,
            close_k=args.texture_close_k,
            blur_ksize=args.texture_blur_ksize,
        )
        # Overlay: paint green where edges are present
        texture_overlay = img.copy()
        texture_overlay[tex_edges > 0] = (
            0.3 * texture_overlay[tex_edges > 0] + np.array([0, 255, 0]) * 0.7
        ).astype(np.uint8)
    else:
        tex_edges = None
        texture_overlay = None

    # Save files
    mask_path = out_dir / (input_path.stem + "_mask.png")
    overlay_path = out_dir / (input_path.stem + "_overlay.png")
    svg_path = out_dir / (input_path.stem + "_contours.svg")
    tex_edges_path = out_dir / (input_path.stem + "_texture_edges.png")
    tex_overlay_path = out_dir / (input_path.stem + "_texture_overlay.png")

    cv.imwrite(str(mask_path), bin_img)
    cv.imwrite(str(overlay_path), overlay)
    if args.save_svg:
        save_svg(contours, svg_path, width=w, height=h)

    print(f"Saved: {mask_path}")
    print(f"Saved: {overlay_path}")
    if args.save_svg:
        print(f"Saved: {svg_path}")
    if args.texture:
        cv.imwrite(str(tex_edges_path), tex_edges)
        cv.imwrite(str(tex_overlay_path), texture_overlay)
        print(f"Saved: {tex_edges_path}")
        print(f"Saved: {tex_overlay_path}")
    print(f"Contours: {len(contours)} | Image: {w}x{h} | Fill: {np.mean(bin_img>0):.1%}")


if __name__ == '__main__':
    # 1) Pre-parse to get --config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, help='Path to TOML config file (values can be overridden by CLI)')
    pre_args, remaining_argv = pre.parse_known_args()

    cfg_defaults = {}
    # Determine config path: explicit > auto-detect > None
    cfg_path: Path | None = None
    if pre_args.config:
        cfg_path = Path(pre_args.config)
    else:
        # Auto-detect common locations
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / 'contours_config.toml',
            Path('process/contours_config.toml'),
            Path('contours_config.toml'),
        ]
        for c in candidates:
            if c.exists():
                cfg_path = c
                break

    if cfg_path is not None:
        if tomllib is None:
            print('tomllib is unavailable. Use Python 3.11+ or remove --config.', file=sys.stderr)
            sys.exit(3)
        if not cfg_path.exists():
            print(f'Config not found: {cfg_path}', file=sys.stderr)
            sys.exit(3)
        with open(cfg_path, 'rb') as f:
            cfg = tomllib.load(f)
        if not isinstance(cfg, dict):
            print('Invalid config format (expected TOML table).', file=sys.stderr)
            sys.exit(3)
        cfg_defaults = cfg

    # 2) Build main parser and apply defaults from config, then parse remaining CLI
    parser = argparse.ArgumentParser(description='Extract contours from an image', parents=[pre])
    parser.add_argument('--input', type=str, default='process/demo.png')
    parser.add_argument('--output-dir', type=str, default='process/out')
    parser.add_argument('--method', type=str, default='auto', choices=['auto','otsu','adaptive','canny'])
    parser.add_argument('--invert', type=str, default='auto', choices=['auto','true','false'])
    parser.add_argument('--min-area', type=float, default=0.005)
    parser.add_argument('--morph', action='store_true', default=True)
    parser.add_argument('--close-k', type=int, default=7)
    parser.add_argument('--open-k', type=int, default=5)
    parser.add_argument('--close-iter', type=int, default=2)
    parser.add_argument('--open-iter', type=int, default=1)
    parser.add_argument('--external', action='store_true', help='only external contours')
    parser.add_argument('--keep-biggest', action='store_true', help='keep largest connected component')
    parser.add_argument('--simplify', type=float, default=0.0, help='RDP epsilon ratio; 0 to disable (default 0 for maximum fidelity)')
    parser.add_argument('--chain', type=str, default='none', choices=['none','simple','tc89','tc89_l1'], help='Contour chain approximation (default none for maximum fidelity)')
    parser.add_argument('--thickness', type=int, default=2, help='Overlay polyline thickness')
    parser.add_argument('--bilateral', action='store_true', default=True, help='use bilateral filter instead of Gaussian')
    parser.add_argument('--save-svg', action='store_true')

    # Enhancement controls
    parser.add_argument('--enhance', action='store_true', default=True, help='apply image enhancement (CLAHE + saturation + gamma)')
    parser.add_argument('--clahe-clip', type=float, default=4.0)
    parser.add_argument('--clahe-tile', type=int, default=8)
    parser.add_argument('--sat-factor', type=float, default=1.2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--use-enhanced-for-mask', action='store_true', default=True, help='use enhanced image for mask extraction')

    # Texture edge extraction
    parser.add_argument('--texture', action='store_true', default=True, help='enable texture edges output')
    parser.add_argument('--texture-method', type=str, default='canny', choices=['canny','sobel','laplacian','fusion'])
    parser.add_argument('--texture-sigma', type=float, default=0.33)
    parser.add_argument('--texture-sobel-ksize', type=int, default=3)
    parser.add_argument('--texture-lap-ksize', type=int, default=3)
    parser.add_argument('--texture-thresh', default='auto', help='threshold for Sobel/Laplacian magnitude; number or "auto"')
    parser.add_argument('--texture-close-k', type=int, default=3)
    parser.add_argument('--texture-blur-ksize', type=int, default=3)
    parser.add_argument('--use-enhanced-for-texture', action='store_true', default=True, help='use enhanced image for texture extraction')

    if cfg_defaults:
        # argparse will use these as defaults, and CLI values in remaining_argv will override
        # Convert TOML keys (already snake_case) to match arg dest names
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args(remaining_argv)
    run(args)
