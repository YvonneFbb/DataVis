#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal texture edge extraction tool.
Outputs:
- <name>_texture_edges.png
- <name>_texture_overlay.png

Config: TOML via --config (see process/texture_config.toml)
CLI overrides config values.
"""
import argparse
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
try:
    import tomllib  # py311+
except Exception:
    tomllib = None


def apply_clahe_y(image, clip_limit=4.0, tile_grid=8):
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    y_clahe = clahe.apply(y)
    out = cv.merge((y_clahe, cr, cb))
    return cv.cvtColor(out, cv.COLOR_YCrCb2BGR)


def adjust_saturation(image, factor=1.2):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv.split(hsv)
    s *= float(factor)
    s = np.clip(s, 0, 255)
    hsv_enhanced = cv.merge((h, s, v)).astype(np.uint8)
    return cv.cvtColor(hsv_enhanced, cv.COLOR_HSV2BGR)


def apply_gamma(image, gamma=1.0):
    if gamma is None or abs(gamma - 1.0) < 1e-3:
        return image
    inv = 1.0 / float(gamma)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype('uint8')
    return cv.LUT(image, table)


def enhance_image(image, enable=True, clahe_clip=4.0, clahe_tile=8, sat_factor=1.2, gamma=1.0):
    if not enable:
        return image
    out = apply_clahe_y(image, clip_limit=clahe_clip, tile_grid=clahe_tile)
    out = adjust_saturation(out, factor=sat_factor)
    out = apply_gamma(out, gamma=gamma)
    return out


def edges_canny_auto(gray, sigma=0.33):
    v = float(np.median(gray))
    lower = max(0, (1.0 - sigma) * v)
    upper = min(255, (1.0 + sigma) * v)
    return cv.Canny(gray, lower, upper)


def edges_sobel(gray, ksize=3, thresh='auto'):
    sx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    sy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    mag = cv.magnitude(sx, sy)
    mag_u8 = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if thresh == 'auto':
        _, edges = cv.threshold(mag_u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        _, edges = cv.threshold(mag_u8, float(thresh), 255, cv.THRESH_BINARY)
    return edges


def edges_laplacian(gray, ksize=3, thresh='auto'):
    lap = cv.Laplacian(gray, cv.CV_32F, ksize=ksize)
    mag = np.abs(lap)
    mag_u8 = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if thresh == 'auto':
        _, edges = cv.threshold(mag_u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        _, edges = cv.threshold(mag_u8, float(thresh), 255, cv.THRESH_BINARY)
    return edges


def extract_texture_edges(image,
                          method='canny',
                          sigma=0.33,
                          sobel_ksize=3,
                          lap_ksize=3,
                          thresh='auto',
                          close_k=3,
                          blur_ksize=3):
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
    return (edges > 0).astype(np.uint8) * 255


def main(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv.imread(str(input_path))
    if img is None:
        print("Failed to read image.")
        sys.exit(1)

    # Enhance
    img_enh = enhance_image(
        img,
        enable=args.enhance,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        sat_factor=args.sat_factor,
        gamma=args.gamma,
    )

    src = img_enh if args.use_enhanced else img
    tex_edges = extract_texture_edges(
        src,
        method=args.texture_method,
        sigma=args.texture_sigma,
        sobel_ksize=args.texture_sobel_ksize,
        lap_ksize=args.texture_lap_ksize,
        thresh=args.texture_thresh,
        close_k=args.texture_close_k,
        blur_ksize=args.texture_blur_ksize,
    )

    texture_overlay = img.copy()
    texture_overlay[tex_edges > 0] = (
        0.3 * texture_overlay[tex_edges > 0] + np.array([0, 255, 0]) * 0.7
    ).astype(np.uint8)

    edges_path = out_dir / (input_path.stem + "_texture_edges.png")
    overlay_path = out_dir / (input_path.stem + "_texture_overlay.png")
    cv.imwrite(str(edges_path), tex_edges)
    cv.imwrite(str(overlay_path), texture_overlay)
    print(f"Saved: {edges_path}")
    print(f"Saved: {overlay_path}")


if __name__ == '__main__':
    # pre-parse config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str)
    pre_args, remaining = pre.parse_known_args()

    cfg_defaults = {}
    # Determine config path: explicit > auto-detect > None
    cfg_path = None
    if pre_args.config:
        cfg_path = Path(pre_args.config)
    else:
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / 'texture_config.toml',
            Path('process/texture_config.toml'),
            Path('texture_config.toml'),
        ]
        for c in candidates:
            if c.exists():
                cfg_path = c
                break
    if cfg_path is not None:
        if tomllib is None:
            print('tomllib unavailable. Use Python 3.11+.', file=sys.stderr)
            sys.exit(3)
        if not cfg_path.exists():
            print(f'Config not found: {cfg_path}', file=sys.stderr)
            sys.exit(3)
        with open(cfg_path, 'rb') as f:
            cfg_defaults = tomllib.load(f)

    parser = argparse.ArgumentParser(parents=[pre], description='Texture edge extraction')
    parser.add_argument('--input', type=str, default='process/demo.png')
    parser.add_argument('--output-dir', type=str, default='process/out')
    # enhancement
    parser.add_argument('--enhance', action='store_true', default=True)
    parser.add_argument('--clahe-clip', type=float, default=4.0)
    parser.add_argument('--clahe-tile', type=int, default=8)
    parser.add_argument('--sat-factor', type=float, default=1.2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--use-enhanced', action='store_true', default=True)
    # texture method
    parser.add_argument('--texture-method', type=str, default='fusion', choices=['canny','sobel','laplacian','fusion'])
    parser.add_argument('--texture-sigma', type=float, default=0.33)
    parser.add_argument('--texture-sobel-ksize', type=int, default=3)
    parser.add_argument('--texture-lap-ksize', type=int, default=3)
    parser.add_argument('--texture-thresh', default='auto')
    parser.add_argument('--texture-close-k', type=int, default=3)
    parser.add_argument('--texture-blur-ksize', type=int, default=3)

    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args(remaining)
    main(args)
