#!/usr/bin/env python3
"""
preOCR 批处理脚本：调用远程 PaddleOCR 服务，对输入图片(单张或目录)进行文本区域检测，
将结果写入 data/results/preocr/<basename>/ 下的 regions.json 与 region_images/。

依赖: RemotePaddleOCRClient 提供的 predict_full_document 接口（如果没有，可先用 predict_single_character 近似）。
当前 remote_paddleocr.py 未实现 full document 返回 text_regions 的结构时，会回退逐字符模拟（在整图上调用 predict_single_character 仅生成一个区域）。
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src.config import PREOCR_DIR
except Exception as e:
    raise RuntimeError(f"无法导入 config: {e}")

from .remote_paddleocr import RemotePaddleOCRClient
import cv2
import numpy as np

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def _list_images(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in IMG_EXT:
        return [p]
    if p.is_dir():
        return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT])
    raise FileNotFoundError(path)


def run_preocr_on_image(client: RemotePaddleOCRClient, image_path: Path) -> Dict[str, Any]:
    # 尝试 full document
    try:
        if hasattr(client, 'predict_full_document'):
            res = client.predict_full_document(image_path)
        else:
            res = {'success': False, 'message': 'no predict_full_document'}
    except Exception as e:
        res = {'success': False, 'message': str(e)}

    img = cv2.imread(str(image_path))
    if img is None:
        return {'success': False, 'error': 'cannot read image'}

    base = image_path.stem
    out_dir = Path(PREOCR_DIR) / base
    region_dir = out_dir / 'region_images'
    region_dir.mkdir(parents=True, exist_ok=True)

    regions: List[Dict[str, Any]] = []
    overlay = img.copy()
    # 预定义一组可区分颜色 (BGR)
    palette = [
        (0,255,0),      # 1 绿色
        (0,128,255),    # 2 橙色
        (255,0,0),      # 3 蓝色
        (0,0,255),      # 4 红色
        (255,0,255),    # 5 洋红
        (255,255,0),    # 6 青黄
        (128,0,255),    # 7 紫
        (0,255,255),    # 8 黄
        (180,0,128),    # 9 暗紫
        (0,180,180),    # 10 暗青
    ]

    if res.get('success') and res.get('text_regions'):
        for i, r in enumerate(res['text_regions'], 1):
            # bbox 可能是 [x1,y1,x2,y2] 或 [[x,y], ...] 多边形
            bbox_raw = r.get('bbox') or r.get('rect') or r.get('box')
            x1=y1=0; x2=img.shape[1]; y2=img.shape[0]
            poly_pts = None
            if bbox_raw:
                try:
                    # 扁平简单形式
                    if all(isinstance(v,(int,float)) for v in bbox_raw[:4]):
                        x1,y1,x2,y2 = map(int, bbox_raw[:4])
                    else:
                        # 可能是点列表
                        pts = []
                        for pt in bbox_raw:
                            if isinstance(pt,(list,tuple)) and len(pt)>=2 and all(isinstance(c,(int,float)) for c in pt[:2]):
                                pts.append((float(pt[0]), float(pt[1])))
                        if pts:
                            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                            x1=int(min(xs)); x2=int(max(xs)); y1=int(min(ys)); y2=int(max(ys))
                            poly_pts = pts
                except Exception:
                    pass
            # 合法性裁剪
            h_img, w_img = img.shape[:2]
            x1=max(0,min(x1,w_img-1)); x2=max(x1+1,min(x2,w_img))
            y1=max(0,min(y1,h_img-1)); y2=max(y1+1,min(y2,h_img))
            crop = img[y1:y2, x1:x2]
            region_name = f'region_{i:03d}.jpg'
            cv2.imwrite(str(region_dir / region_name), crop)
            regions.append({
                'id': i,
                'file': region_name,
                'bbox': [x1,y1,x2,y2],
                'text': r.get('text',''),
                'confidence': r.get('confidence', 0.0)
            })
            # 画框 / 多边形
            color = palette[(i-1) % len(palette)]
            if poly_pts and len(poly_pts) >= 3:
                pts_i = [(int(px), int(py)) for px,py in poly_pts]
                cv2.polylines(overlay, [cv2.convexHull(np.array(pts_i, dtype=np.int32))], True, color, 1)
            else:
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 1)
            # 仅显示编号，避免 OpenCV 字库不支持汉字而显示 '?'
            label = str(i)
            cv2.putText(overlay, label, (x1, max(12,y1+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:
        # 回退: 用单字符预测模拟一个区域（整图）
        h, w = img.shape[:2]
        regions.append({'id': 1, 'file': 'region_001.jpg', 'bbox': [0,0,w,h], 'text': '', 'confidence': 0.0})
        cv2.imwrite(str(region_dir / 'region_001.jpg'), img)
        cv2.rectangle(overlay, (0,0), (w-1,h-1), (0,255,0), 1)
        cv2.putText(overlay, '1', (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    with open(out_dir / 'regions.json', 'w', encoding='utf-8') as f:
        json.dump({'image': str(image_path), 'regions': regions}, f, ensure_ascii=False, indent=2)
    # 保存 overlay
    try:
        cv2.imwrite(str(out_dir / 'overlay.jpg'), overlay)
    except Exception:
        pass
    return {'success': True, 'regions': len(regions), 'out_dir': str(out_dir), 'overlay': str(out_dir / 'overlay.jpg')}


def run_preocr(input_path: str) -> Dict[str, Any]:
    client = RemotePaddleOCRClient()
    images = _list_images(input_path)
    summary = []
    for img in images:
        try:
            res = run_preocr_on_image(client, img)
            summary.append(res)
        except Exception as e:
            summary.append({'success': False, 'error': str(e), 'image': str(img)})
    return {'total': len(images), 'results': summary}

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Run remote preOCR to extract regions')
    ap.add_argument('input', help='单张图片或目录')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    out = run_preocr(args.input)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out)
