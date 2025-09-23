"""Unified project configuration (整理版)

按流水线阶段划分：
1. 路径 / 目录
2. 环境变量辅助
3. PREPROCESS
4. PREOCR
5. SEGMENT
6. POSTOCR
7. 校验与摘要工具
"""

import os
from typing import Dict, Any

# ==================== 1. 路径 / 目录 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, 'preprocessed')
SEGMENTS_DIR = os.path.join(RESULTS_DIR, 'segments')
OCR_DIR = os.path.join(RESULTS_DIR, 'ocr')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')
PREOCR_DIR = os.path.join(RESULTS_DIR, 'preocr')  # 预OCR（文本区域检测）目录

# ==================== 2. 环境变量辅助 ====================
def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

# ==================== 3. PREPROCESS (图像预处理) ====================
PREPROCESS_STROKE_HEAL_CONFIG = {
    'enabled': False,
    'kernel': 3,
    'iterations': 1,
    'directions': ['iso', 'h', 'v'],
    'bilateral_denoise': False,
    'bilateral': {
        'd': 5,
        'sigmaColor': 30,
        'sigmaSpace': 15,
    },
}

PREPROCESS_INK_PRESERVE_CONFIG = {
    'enabled': True,
    'blackhat_kernel': 9,
    'blackhat_strength': 0.6,
    'unsharp_amount': 0.2,
}

# ==================== 4. PREOCR (区域检测 / 远程 OCR) ====================
OCR_REMOTE_CONFIG = {
    'server_url': _env('PPOCR_SERVER_URL', 'http://172.16.1.154:8000'),
    'timeout': _env_int('PPOCR_TIMEOUT', 30),
}

# ==================== 5. SEGMENT (连通域 + 投影裁切) ====================
SEGMENT_REFINE_CONFIG = {
    'enabled': True,
    'mode': 'ccprojection',
    'expand_px': 4,
    'final_pad': 0,
    'debug_visualize': True,
    'debug_dirname': 'debug',
}

PROJECTION_TRIM_CONFIG = {
    'binarize': 'otsu',
    'adaptive_block': 31,
    'adaptive_C': 3,
    'run_min_coverage_ratio': 0.01,
    'run_min_coverage_abs': 0.005,
    'primary_run_min_mass_ratio': 0.5,
    'primary_run_min_length_ratio': 0.3,
    'tighten_min_coverage': 0.01,
    'horizontal_trim_limit_ratio': 0.2,
    'horizontal_trim_limit_px': 0,
    'vertical_trim_limit_ratio': 0.2,
    'vertical_trim_limit_px': 0,
}

CC_FILTER_CONFIG = {
    'edge_margin': 2,
    'min_area_ratio': 0.02,
    'max_aspect_for_edge': 6.0,
    'min_dim_px': 2,
    'border_touch_min_area_ratio': 0.003,
}

# ==================== 6. POSTOCR (OCR 过滤) ====================
OCR_FILTER_CONFIG = {
    'enabled': True,
    'confidence_threshold': 0.3,
    'min_char_width': 8,
    'min_char_height': 8,
    'language_preference': ['zh-Hans', 'zh-Hant'],
    'framework': 'accurate',
    'max_text_length': 5,
    'allow_empty': False,
    'chinese_only': False,
    'batch_size': 50,
}

# ==================== 8. 校验与摘要工具 ====================
def validate_config() -> None:
    if SEGMENT_REFINE_CONFIG.get('expand_px', 0) < 0:
        raise ValueError('SEGMENT_REFINE_CONFIG.expand_px 不能为负数')
    for key in ('horizontal_trim_limit_ratio', 'vertical_trim_limit_ratio'):
        if PROJECTION_TRIM_CONFIG.get(key, 1.0) < 0:
            raise ValueError(f'PROJECTION_TRIM_CONFIG.{key} 不能为负数')
    for key in ('horizontal_trim_limit_px', 'vertical_trim_limit_px'):
        if PROJECTION_TRIM_CONFIG.get(key, 0) < 0:
            raise ValueError(f'PROJECTION_TRIM_CONFIG.{key} 不能为负数')

def config_summary(compact: bool = True) -> Dict[str, Any]:
    summary = {
        'paths': {
            'PROJECT_ROOT': PROJECT_ROOT,
            'DATA_DIR': DATA_DIR,
            'RESULTS_DIR': RESULTS_DIR,
        },
        'preprocess': {
            'stroke_heal': PREPROCESS_STROKE_HEAL_CONFIG,
            'ink_preserve': PREPROCESS_INK_PRESERVE_CONFIG,
        },
        'preocr': OCR_REMOTE_CONFIG,
        'segment': {
            'refine': SEGMENT_REFINE_CONFIG,
            'projection_trim': PROJECTION_TRIM_CONFIG,
            'cc_filter': CC_FILTER_CONFIG,
        },
        'postocr': {
            'ocr_filter': OCR_FILTER_CONFIG,
        },
    }
    if compact:
        return summary
    return {
        'PREPROCESS_STROKE_HEAL_CONFIG': PREPROCESS_STROKE_HEAL_CONFIG,
        'PREPROCESS_INK_PRESERVE_CONFIG': PREPROCESS_INK_PRESERVE_CONFIG,
        'SEGMENT_REFINE_CONFIG': SEGMENT_REFINE_CONFIG,
        'PROJECTION_TRIM_CONFIG': PROJECTION_TRIM_CONFIG,
        'OCR_FILTER_CONFIG': OCR_FILTER_CONFIG,
        'OCR_REMOTE_CONFIG': OCR_REMOTE_CONFIG,
    }

__all__ = [
    'PROJECT_ROOT','DATA_DIR','RAW_DIR','RESULTS_DIR','PREPROCESSED_DIR','SEGMENTS_DIR','OCR_DIR','ANALYSIS_DIR','PREOCR_DIR',
    'PREPROCESS_STROKE_HEAL_CONFIG','PREPROCESS_INK_PRESERVE_CONFIG',
    'SEGMENT_REFINE_CONFIG','PROJECTION_TRIM_CONFIG','CC_FILTER_CONFIG',
    'OCR_FILTER_CONFIG','OCR_REMOTE_CONFIG',
    'validate_config','config_summary'
]
