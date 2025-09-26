from __future__ import annotations

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
PREOCR_DIR = os.path.join(RESULTS_DIR, 'preocr')              # 预OCR（文本区域检测）
SEGMENTS_DIR = os.path.join(RESULTS_DIR, 'segments')          # 分割结果
POSTOCR_DIR = os.path.join(RESULTS_DIR, 'postocr')            # 大模型过滤结果
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')

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
    'enabled': False,            # 是否启用断笔修补
    'kernel': 3,                 # 闭运算核尺寸
    'iterations': 1,             # 闭运算次数
    'directions': ['iso', 'h', 'v'],  # 方向枚举
    'bilateral_denoise': False,  # 是否在修补前使用双边滤波
    'bilateral': {
        'd': 5,                 # 双边滤波像素邻域
        'sigmaColor': 30,       # 双边滤波颜色参数
        'sigmaSpace': 15,       # 双边滤波空间参数
    },
}

PREPROCESS_INK_PRESERVE_CONFIG = {
    'enabled': True,            # 是否启用回墨增强
    'blackhat_kernel': 9,       # 黑帽核尺寸
    'blackhat_strength': 0.6,   # 黑帽增强强度
    'unsharp_amount': 0.2,      # 反锐化系数
}

# ==================== 4. PREOCR (区域检测 / 远程 OCR) ====================
OCR_REMOTE_CONFIG = {
    'server_url': _env('PPOCR_SERVER_URL', 'http://172.16.1.154:8000'), # PaddleOCR 服务地址
    'timeout': _env_int('PPOCR_TIMEOUT', 30),                            # 请求超时时间
}

# ==================== 5. SEGMENT (连通域 + 投影裁切) ====================
SEGMENT_REFINE_CONFIG = {
    'enabled': True,            # 是否启用分割精修
    'mode': 'ccprojection',     # 模式：ccprojection / projection_only / cc_debug
    'expand_px': {              # ROI 额外扩张像素（分别对应左右上下）
        'left': 2,
        'right': 2,
        'top': 4,
        'bottom': 0,
    },
    'final_pad': 0,             # 裁剪后的统一回填像素
    'debug_visualize': True,    # 是否输出调试图
    'debug_dirname': 'debug',   # 调试图目录名称
}

PROJECTION_TRIM_CONFIG = {
    'binarize': 'otsu',                 # 投影前的二值化方式
    'adaptive_block': 31,               # 自适应阈值块大小
    'adaptive_C': 3,                    # 自适应阈值常数
    'run_min_coverage_ratio': 0.01,     # run 判定：覆盖度占最大值的比例
    'run_min_coverage_abs': 0.005,      # run 判定：覆盖度绝对下限
    'primary_run_min_mass_ratio': 0.5,  # 主 run 需占投影质量的比例
    'primary_run_min_length_ratio': 0.3,# 主 run 需占宽度的比例
    'tighten_min_coverage': 0.01,       # 主 run 内重新贴边的阈值
    'horizontal_trim_limit_ratio': 0.25,# 左右最大可裁比例（<=0 表示不限）
    'horizontal_trim_limit_px': 0,      # 左右最大可裁像素
    'vertical_trim_limit_ratio': 0.3,  # 上下最大可裁比例
    'vertical_trim_limit_px': 0,        # 上下最大可裁像素
}

CC_FILTER_CONFIG = {
    'border_touch_margin': 1,           # 实际触边判定范围（像素）
    'edge_zone_margin': 2,              # 边缘区域判定范围（像素）
    'border_touch_min_area_ratio': 0.01,  # 触边组件最小面积比例
    'edge_zone_min_area_ratio': 0.01,     # 边缘区域组件最小面积比例
    'interior_min_area_ratio': 0.005,     # 内部组件最小面积比例
    'max_aspect_for_edge': 6.0,         # 边缘/触边组件最大长宽比
    'min_dim_px': 4,                    # 边缘/触边组件最小尺寸（像素）
}

EDGE_BREAKING_CONFIG = {
    'enabled': True,                    # 是否启用边缘薄连接断开
    'method': 'intelligent',            # 断连方法：intelligent / selective
    'min_aspect_ratio': 3.0,            # 连通域长宽比阈值（>=此值才考虑断开）
    'kernel_size': 2,                   # 形态学开运算核尺寸
    'min_remaining_area': 10,           # 断开后剩余组件的最小面积
    'edge_margin': 2,                   # 边缘判定的缓冲像素
    'edge_ratio': 0.15,                 # 选择性方法：边缘区域占比
}

NOISE_REMOVAL_CONFIG = {
    'enabled': True,                   # 是否启用杂质色块清理
    'dark_stroke_threshold': 60,       # 深色笔画阈值（<=此值认为是文字主体）
    'light_noise_threshold': 230,      # 淡色杂质阈值（>深色且<此值的区域为杂质候选）
    'min_noise_area': 3,               # 最小杂质面积
    'max_noise_area': 300,             # 最大杂质面积（绝对像素数）
    'large_noise_area_threshold': 50,  # 大块杂质面积阈值（超过此值认为是杂质而非文字边缘）
}

# ==================== 6. POSTOCR (质量过滤) ====================
ARK_VISION_CONFIG = {
    'enabled': True,                       # 是否启用大模型过滤
    'base_url': 'https://ark.cn-beijing.volces.com/api/v3', # Ark API 地址
    'model': 'doubao-seed-1-6-vision-250815',               # 模型名称
    'api_key_env': 'ARK_API_KEY',          # API Key 环境变量
    'timeout': 60,                         # 请求超时时间
    'temperature': 0.0,                    # 采样温度
    'max_tokens': 256,                     # 最大返回 token 数
    'workers': 16,                          # 并发处理的目录数
}

# ==================== 8. 校验与摘要工具 ====================
def validate_config() -> None:
    expand_cfg = SEGMENT_REFINE_CONFIG.get('expand_px', 0)
    if isinstance(expand_cfg, dict):
        for key in ('left', 'right', 'top', 'bottom'):
            val = expand_cfg.get(key, 0)
            if val < 0:
                raise ValueError(f'SEGMENT_REFINE_CONFIG.expand_px.{key} 不能为负数')
    else:
        if expand_cfg < 0:
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
            'POSTOCR_DIR': POSTOCR_DIR,
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
            'edge_breaking': EDGE_BREAKING_CONFIG,
            'noise_removal': NOISE_REMOVAL_CONFIG,
        },
        'postocr': {
            'ark_model': ARK_VISION_CONFIG,
        },
    }
    if compact:
        return summary
    return {
        'PREPROCESS_STROKE_HEAL_CONFIG': PREPROCESS_STROKE_HEAL_CONFIG,
        'PREPROCESS_INK_PRESERVE_CONFIG': PREPROCESS_INK_PRESERVE_CONFIG,
        'SEGMENT_REFINE_CONFIG': SEGMENT_REFINE_CONFIG,
        'PROJECTION_TRIM_CONFIG': PROJECTION_TRIM_CONFIG,
        'CC_FILTER_CONFIG': CC_FILTER_CONFIG,
        'EDGE_BREAKING_CONFIG': EDGE_BREAKING_CONFIG,
        'NOISE_REMOVAL_CONFIG': NOISE_REMOVAL_CONFIG,
        'OCR_REMOTE_CONFIG': OCR_REMOTE_CONFIG,
        'ARK_VISION_CONFIG': ARK_VISION_CONFIG,
    }

__all__ = [
    'PROJECT_ROOT','DATA_DIR','RAW_DIR','RESULTS_DIR','PREPROCESSED_DIR','SEGMENTS_DIR','POSTOCR_DIR','ANALYSIS_DIR','PREOCR_DIR',
    'PREPROCESS_STROKE_HEAL_CONFIG','PREPROCESS_INK_PRESERVE_CONFIG',
    'SEGMENT_REFINE_CONFIG','PROJECTION_TRIM_CONFIG','CC_FILTER_CONFIG','EDGE_BREAKING_CONFIG','NOISE_REMOVAL_CONFIG',
    'OCR_REMOTE_CONFIG','ARK_VISION_CONFIG',
    'validate_config','config_summary'
]
