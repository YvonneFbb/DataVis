"""Unified project configuration (整理版)

层次结构:
1. 路径路径 / 目录
2. 环境变量读取辅助 (_env, _env_int)
3. 预处理配置 (断笔修补 / 墨色保持)
4. 分割细化配置
5. OCR 过滤 & 远程服务
6. VL 相关配置与目录
7. 裁边 / 噪点清理
8. 校验与摘要工具

注意：外部模块引用的变量名称保持不变，避免破坏现有 import。
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

# ==================== 预处理：断笔/小缺口修补（可选） ====================
# 在送入 OCR 与分割前，对灰度图进行“逆向闭运算”来弥合细小裂缝/断笔，
# 有助于后续自适应阈值时保持连贯的笔画结构。
PREPROCESS_STROKE_HEAL_CONFIG = {
    'enabled': False,            # 默认关闭，避免影响现有效果；需要时再开启
    'kernel': 3,                 # 结构元素的基准尺寸（像素，建议 3 或 5）
    'iterations': 1,             # 闭运算迭代次数
    # 方向：iso(各向同性圆核)、h(水平)、v(垂直)、d1(↘ 对角)、d2(↙ 对角)
    'directions': ['iso', 'h', 'v'],
    # 可选：在闭运算前做一点双边滤波，平滑纸张纹理、保留边缘
    'bilateral_denoise': False,
    'bilateral': {
        'd': 5,
        'sigmaColor': 30,
        'sigmaSpace': 15,
    },
}

# ==================== 预处理：墨色保持/增强（避免变淡） ====================
# 在整体亮度提增后，对文字墨迹进行“回墨”处理，避免笔画显得发灰变淡。
PREPROCESS_INK_PRESERVE_CONFIG = {
    'enabled': True,        # 默认开启，尽量避免整体变淡
    'blackhat_kernel': 9,   # 黑帽核尺寸（奇数，建议 7/9/11）
    'blackhat_strength': 0.6,  # 回墨强度（0~1）
    'unsharp_amount': 0.2,  # 反锐化增强边缘的权重（0 关闭）
}

# ==================== 分割细化与对齐（当前实现使用） ====================

# 分割细化与对齐（动态 seam + 期望数量对齐）
SEGMENT_REFINEMENT_CONFIG = {
    # 是否启用动态 seam 切割（在初步分割的谷线附近寻找一条“避开笔画”的最小代价路径来切割）
    'enable_dynamic_seam': True,
    # seam 搜索带宽（相对于估计单字高度的比例；例如 0.3 表示在 ±0.3H 的窗口内搜索路径）
    'seam_band_ratio': 0.1,
    # seam 代价权重：越小越偏好路径通过该特征
    'seam_ink_weight': 1.0,   # 墨迹（前景）代价
    'seam_dist_weight': 0.5,  # 距离前景越远越好（使用背景距前景的DT）
    'seam_grad_weight': 0.2,  # 图像梯度，惩罚切过强边缘
    # 期望数量对齐（利用外部提供的期望字数进行轻量 split/merge 调整）
    'expected_count_alignment': True,
    # 允许的最小分段高度（相对整图高度），避免产生过薄的小碎片
    'min_segment_height_ratio': 0.25,
    'max_split_attempts': 3,
    'max_merge_attempts': 3,
    # 调试叠加绘制
    'debug_overlay': True,
}

#（以下 legacy 配置已移除：IQR_MULTIPLIER、VERTICAL_CHAR_CONFIG、WIDE_CHAR_CONFIG、
# RECLASSIFY_CONFIG、PROJECTION_CONFIG、RED_RED_MERGE_CONFIG、RED_GREEN_MERGE_CONFIG、
# CHAR_CLASSIFICATION_CONFIG。如需回退旧算法，可从版本历史中恢复。）

# OCR过滤配置
OCR_FILTER_CONFIG = {
    'enabled': True,                    # 是否启用OCR过滤
    'confidence_threshold': 0.3,        # 最小置信度阈值
    'min_char_width': 8,               # 最小字符宽度（像素）
    'min_char_height': 8,              # 最小字符高度（像素）  
    'language_preference': ['zh-Hans', 'zh-Hant'],  # 语言偏好
    'framework': 'accurate',            # OCR框架: 'accurate', 'fast', 'livetext'
    'max_text_length': 5,              # 识别文本最大长度（超出认为识别错误）
    'allow_empty': False,              # 是否允许空识别结果
    'chinese_only': False,             # 是否只保留中文字符
    'batch_size': 50,                  # 批处理大小
}

# 远程 PaddleOCR 客户端配置（可由环境变量覆盖）
OCR_REMOTE_CONFIG = {
    'server_url': _env('PPOCR_SERVER_URL', 'http://172.16.1.154:8000'),
    'timeout': _env_int('PPOCR_TIMEOUT', 30),
}

# VL (Vision Language) 模型配置
VL_CONFIG = {
    'enabled': True,                   # 是否启用VL模型
    'provider': 'siliconflow',         # API提供商: 'siliconflow', 'openai', 'anthropic'
    'api_key': os.environ.get('SILICONFLOW_API_KEY', ''),  # API密钥（从环境变量获取）
    'base_url': 'https://api.siliconflow.cn/v1',  # API基础URL
    'model': 'Qwen/Qwen2.5-VL-72B-Instruct',      # 使用的VL模型
    'max_tokens': 100,                 # 最大输出token数
    'temperature': 0.1,                # 生成温度，低温度更准确
    'timeout': 30,                     # 请求超时时间（秒）
    'retry_times': 3,                  # 重试次数
    'batch_size': 10,                  # 批处理大小（API并发限制）
    'image_detail': 'high',            # 图像处理精度: 'low', 'high', 'auto'
    'prompt_template': '这是一张古籍中的单个汉字图片，请仔细识别这个字符。只输出识别出的单个汉字，不要添加任何解释或标点符号。如果无法识别请输出"无法识别"。',
    'fallback_enabled': True,          # 是否启用OCR作为fallback
    'confidence_estimation': True,     # 是否启用置信度估计
}

# VL字符质量评估配置
VL_CHARACTER_EVALUATION_CONFIG = {
    'enabled': False,                  # 是否启用VL字符质量评估（测试简化方案）
    'timeout': 60,                     # 单个字符评估超时时间（秒）
    'batch_size': 5,                   # 批处理大小（避免API限制）
    'rate_limit_delay': 1,             # 批次间延迟（秒）
    'save_results': True,              # 是否保存评估结果
    'filter_mode': 'vl',               # 过滤模式: 'vl', 'ocr', 'both'
    'quality_threshold': ['GOOD'],     # 接受的质量等级
    'max_tokens': 50,                  # 评估响应的最大token数
}

# VL模型相关目录
VL_DIR = os.path.join(RESULTS_DIR, 'vl')
VL_SEGMENTS_DIR = os.path.join(VL_DIR, 'segments')          # VL筛选后的分割结果
VL_EVALUATIONS_DIR = os.path.join(VL_DIR, 'evaluations')    # VL评估结果文件
VL_ANNOTATIONS_DIR = os.path.join(VL_DIR, 'annotations')    # VL标注图像

# ==================== 字符切片裁边配置 ====================
# 对分割出的单字图进行内容裁边与统一留白，提升可视化与后续识别效果
CHAR_CROP_CONFIG = {
    'enabled': True,          # 是否启用内容裁边
    'mode': 'content',        # 'content' 或 'margin_only'
    'pad': 2,                 # 裁后再扩展的像素
    'final_padding': 0,       # 统一额外边距
    'square_output': False,   # 是否填充成正方形
    'min_fg_area': 8,         # 最小前景面积阈值
    'binarize': 'otsu',       # 'otsu' | 'adaptive'
}

# ==================== 单字噪点清理配置 ====================
# 在切片上执行连通域过滤，移除小墨点/残片；默认保留最大或最居中的主要前景
CHAR_NOISE_CLEAN_CONFIG = {
    'enabled': False,
    'min_area': 10,
    'min_area_ratio': 0.002,
    'keep_largest': True,
    'center_bias': 0.001,
    'second_keep_ratio': 0.35,
    'morph_open': 0,
    'morph_close': 0,
}

# ==================== 8. 校验与摘要工具 ====================
def validate_config() -> None:
    mode = CHAR_CROP_CONFIG.get('mode')
    if mode not in ('content', 'margin_only'):
        raise ValueError(f"CHAR_CROP_CONFIG.mode 非法: {mode}")
    if CHAR_CROP_CONFIG.get('pad', 0) < 0:
        raise ValueError("CHAR_CROP_CONFIG.pad 不能为负数")
    if CHAR_NOISE_CLEAN_CONFIG.get('min_area', 1) < 0:
        raise ValueError("CHAR_NOISE_CLEAN_CONFIG.min_area 不能为负数")
    if SEGMENT_REFINEMENT_CONFIG.get('seam_band_ratio', 0.1) <= 0:
        raise ValueError("SEGMENT_REFINEMENT_CONFIG.seam_band_ratio 应 > 0")

def config_summary(compact: bool = True) -> Dict[str, Any]:
    summary = {
        'paths': {
            'PROJECT_ROOT': PROJECT_ROOT,
            'DATA_DIR': DATA_DIR,
            'RESULTS_DIR': RESULTS_DIR,
        },
        'preprocess': {
            'stroke_heal_enabled': PREPROCESS_STROKE_HEAL_CONFIG.get('enabled'),
            'ink_preserve_enabled': PREPROCESS_INK_PRESERVE_CONFIG.get('enabled'),
        },
        'segmentation': {
            'dynamic_seam': SEGMENT_REFINEMENT_CONFIG.get('enable_dynamic_seam'),
            'expected_count_alignment': SEGMENT_REFINEMENT_CONFIG.get('expected_count_alignment'),
        },
        'crop': {
            'enabled': CHAR_CROP_CONFIG.get('enabled'),
            'mode': CHAR_CROP_CONFIG.get('mode'),
            'pad': CHAR_CROP_CONFIG.get('pad'),
        },
        'noise_clean': {
            'enabled': CHAR_NOISE_CLEAN_CONFIG.get('enabled'),
            'min_area': CHAR_NOISE_CLEAN_CONFIG.get('min_area'),
        },
        'ocr_remote': {
            'server_url': OCR_REMOTE_CONFIG.get('server_url'),
            'timeout': OCR_REMOTE_CONFIG.get('timeout'),
        },
        'vl': {
            'enabled': VL_CONFIG.get('enabled'),
            'model': VL_CONFIG.get('model'),
        }
    }
    if compact:
        return summary
    return {
        'PREPROCESS_STROKE_HEAL_CONFIG': PREPROCESS_STROKE_HEAL_CONFIG,
        'PREPROCESS_INK_PRESERVE_CONFIG': PREPROCESS_INK_PRESERVE_CONFIG,
        'SEGMENT_REFINEMENT_CONFIG': SEGMENT_REFINEMENT_CONFIG,
        'CHAR_CROP_CONFIG': CHAR_CROP_CONFIG,
        'CHAR_NOISE_CLEAN_CONFIG': CHAR_NOISE_CLEAN_CONFIG,
        'OCR_FILTER_CONFIG': OCR_FILTER_CONFIG,
        'OCR_REMOTE_CONFIG': OCR_REMOTE_CONFIG,
        'VL_CONFIG': VL_CONFIG,
        'VL_CHARACTER_EVALUATION_CONFIG': VL_CHARACTER_EVALUATION_CONFIG,
    }

__all__ = [
    'PROJECT_ROOT','DATA_DIR','RAW_DIR','RESULTS_DIR','PREPROCESSED_DIR','SEGMENTS_DIR','OCR_DIR','ANALYSIS_DIR','PREOCR_DIR',
    'PREPROCESS_STROKE_HEAL_CONFIG','PREPROCESS_INK_PRESERVE_CONFIG',
    'SEGMENT_REFINEMENT_CONFIG','CHAR_CROP_CONFIG','CHAR_NOISE_CLEAN_CONFIG',
    'OCR_FILTER_CONFIG','OCR_REMOTE_CONFIG',
    'VL_CONFIG','VL_CHARACTER_EVALUATION_CONFIG',
    'validate_config','config_summary'
]