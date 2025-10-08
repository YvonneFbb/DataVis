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
        'top': 10,
        'bottom': 2,
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

    # 检测范围参数 - 在多大范围内寻找内容边界
    'detection_range': {
        'left_ratio': 0.3,      # 左侧检测范围占总宽度的比例
        'right_ratio': 0.3,     # 右侧检测范围占总宽度的比例
        'top_ratio': 0.3,       # 上侧检测范围占总高度的比例
        'bottom_ratio': 0.5,    # 下侧检测范围占总高度的比例
    },

    # 切割限制参数 - 最多允许切掉多少内容（不包括空白）
    'cut_limits': {
        'left_max_ratio': 0.25,      # 左侧最大切割比例
        'right_max_ratio': 0.25,     # 右侧最大切割比例
        'top_max_ratio': 0.25,      # 上侧最大切割比例
        'bottom_max_ratio': 0.3,    # 下侧最大切割比例
    },

}

CC_FILTER_CONFIG = {
    'border_touch_margin': 1,           # 实际触边判定范围（像素）
    'edge_zone_margin': 2,              # 边缘区域判定范围（像素）
    'border_touch_min_area_ratio': 0.1, # 触边组件最小面积比例
    'edge_zone_min_area_ratio': 0.01,   # 边缘区域组件最小面积比例
    'interior_min_area_ratio': 0.002,   # 内部组件最小面积比例
    'max_aspect_for_edge': 6.0,         # 边缘/触边组件最大长宽比
    'min_dim_px': 2,                    # 边缘/触边组件最小尺寸（宽度或高度的最小值，像素）
    'interior_min_dim_px': 1,           # 内部组件最小尺寸（宽度或高度的最小值，像素）
}

BORDER_REMOVAL_CONFIG = {
    'enabled': True,                    # 是否启用边框去除
    'max_iterations': 5,                # 最大迭代次数（多次执行以完全去除边框）

    # 水平边框检测参数
    'border_max_width_ratio': 0.15,      # 最大边框宽度占比（左右两侧检测范围）
    'border_threshold_ratio': 0.35,      # 边框检测阈值（相对于最大投影值的比例）

    # 突变检测参数
    'spike_min_length_ratio': 0.02,     # 异常高值段最小长度占检测范围的比例
    'spike_max_length_ratio': 0.1,     # 异常高值段最大长度占检测范围的比例
    'spike_gradient_threshold': 0.4,    # 突变梯度阈值（相对于最大投影值）
    'spike_prominence_ratio': 0.5,      # 突出度阈值（峰值相对于周围的突出程度）
    'edge_tolerance': 3,                # 允许的边缘偏移像素数

    # 垂直边框处理参数 - 使用类似 Proj 的结构化方案
    'vertical_detection_range': {
        'top_ratio': 0.3,       # 上侧检测范围占总高度的比例
        'bottom_ratio': 0.3,    # 下侧检测范围占总高度的比例
    },

    'vertical_cut_limits': {
        'top_max_ratio': 0.2,       # 上侧最大切割比例
        'bottom_max_ratio': 0.2,    # 下侧最大切割比例
    },

    'debug_verbose': True,              # 是否输出详细的border debug图
}

NOISE_REMOVAL_CONFIG = {
    'enabled': True,                   # 是否启用杂质色块清理
    'dark_stroke_threshold': 60,       # 深色笔画阈值（<=此值认为是文字主体）
    'light_noise_threshold': 240,      # 淡色杂质阈值（>深色且<此值的区域为杂质候选）
    'min_noise_area': 2,               # 最小杂质面积
    'max_noise_area': 100000,          # 最大杂质面积（绝对像素数）
    'large_noise_area_threshold': 50,  # 大块杂质面积阈值（超过此值认为是杂质而非文字边缘）
}

# ==================== 6. POSTOCR (质量过滤) ====================
POSTOCR_CONFIG = {
    'enabled': True,                       # 是否启用大模型过滤
    'mode': 'batch',                       # 推理模式: 'realtime' / 'batch'
    'provider': 'qwen',                    # 模型提供商: 'doubao' / 'qwen'
    'workers': 16,                         # 并发处理的目录数（仅 realtime 模式）

    # 批量推理配置（仅 batch 模式）
    'batch': {
        'completion_window': '24h',        # 最长等待时间: 24h-336h (支持 h/d 单位)
        'poll_interval': 60,               # 轮询间隔（秒），建议 60
        'max_requests_per_batch': 50000,   # 单个任务最多请求数
        'max_line_size_mb': 6.0,          # 单行最大 6MB（官方限制）
    },

    # Doubao (字节跳动) 配置
    'doubao': {
        'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
        'model': 'doubao-seed-1-6-vision-250815',
        'api_key_env': 'ARK_API_KEY',
        'timeout': 60,
        'temperature': 0.0,
        'max_tokens': 256,
    },

    # Qwen (阿里云通义千问) 配置
    'qwen': {
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen3-vl-plus',          # 可选: qwen-vl-max-latest / qwen-vl-ocr-latest / qwen3-vl-plus
        'api_key_env': 'DASHSCOPE_API_KEY',
        'timeout': 60,
        'temperature': 0.0,
        'max_tokens': 512,
        'enable_thinking': 'auto',         # 'auto': 自动判断（qwen3-vl-plus 启用）/ True: 强制启用 / False: 强制禁用
        'thinking_budget': 8192,           # 推理模式最大 token 数
    },
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

    # 验证新的参数结构
    detection_range = PROJECTION_TRIM_CONFIG.get('detection_range', {})
    if isinstance(detection_range, dict):
        for key in ('left_ratio', 'right_ratio', 'top_ratio', 'bottom_ratio'):
            val = detection_range.get(key, 0.3)
            if val < 0 or val > 1:
                raise ValueError(f'PROJECTION_TRIM_CONFIG.detection_range.{key} 必须在0-1之间')

    cut_limits = PROJECTION_TRIM_CONFIG.get('cut_limits', {})
    if isinstance(cut_limits, dict):
        for key in ('left_max_ratio', 'right_max_ratio', 'top_max_ratio', 'bottom_max_ratio'):
            val = cut_limits.get(key, 0.2)
            if val < 0 or val > 1:
                raise ValueError(f'PROJECTION_TRIM_CONFIG.cut_limits.{key} 必须在0-1之间')

    # 验证Border垂直参数结构
    border_vertical_detection = BORDER_REMOVAL_CONFIG.get('vertical_detection_range', {})
    if isinstance(border_vertical_detection, dict):
        for key in ('top_ratio', 'bottom_ratio'):
            val = border_vertical_detection.get(key, 0.3)
            if val < 0 or val > 1:
                raise ValueError(f'BORDER_REMOVAL_CONFIG.vertical_detection_range.{key} 必须在0-1之间')

    border_vertical_cuts = BORDER_REMOVAL_CONFIG.get('vertical_cut_limits', {})
    if isinstance(border_vertical_cuts, dict):
        for key in ('top_max_ratio', 'bottom_max_ratio'):
            val = border_vertical_cuts.get(key, 0.1)
            if val < 0 or val > 1:
                raise ValueError(f'BORDER_REMOVAL_CONFIG.vertical_cut_limits.{key} 必须在0-1之间')


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
            'border_removal': BORDER_REMOVAL_CONFIG,
            'noise_removal': NOISE_REMOVAL_CONFIG,
        },
        'postocr': POSTOCR_CONFIG,
    }
    if compact:
        return summary
    return {
        'PREPROCESS_STROKE_HEAL_CONFIG': PREPROCESS_STROKE_HEAL_CONFIG,
        'PREPROCESS_INK_PRESERVE_CONFIG': PREPROCESS_INK_PRESERVE_CONFIG,
        'SEGMENT_REFINE_CONFIG': SEGMENT_REFINE_CONFIG,
        'PROJECTION_TRIM_CONFIG': PROJECTION_TRIM_CONFIG,
        'CC_FILTER_CONFIG': CC_FILTER_CONFIG,
        'BORDER_REMOVAL_CONFIG': BORDER_REMOVAL_CONFIG,
        'NOISE_REMOVAL_CONFIG': NOISE_REMOVAL_CONFIG,
        'OCR_REMOTE_CONFIG': OCR_REMOTE_CONFIG,
        'POSTOCR_CONFIG': POSTOCR_CONFIG,
    }

__all__ = [
    'PROJECT_ROOT','DATA_DIR','RAW_DIR','RESULTS_DIR','PREPROCESSED_DIR','SEGMENTS_DIR','POSTOCR_DIR','ANALYSIS_DIR','PREOCR_DIR',
    'PREPROCESS_STROKE_HEAL_CONFIG','PREPROCESS_INK_PRESERVE_CONFIG',
    'SEGMENT_REFINE_CONFIG','PROJECTION_TRIM_CONFIG','CC_FILTER_CONFIG','BORDER_REMOVAL_CONFIG','NOISE_REMOVAL_CONFIG',
    'OCR_REMOTE_CONFIG','POSTOCR_CONFIG',
    'validate_config','config_summary'
]
