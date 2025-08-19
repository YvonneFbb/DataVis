import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# 结果目录
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, 'preprocessed')
SEGMENTS_DIR = os.path.join(RESULTS_DIR, 'segments')
OCR_DIR = os.path.join(RESULTS_DIR, 'ocr')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')

# ==================== 字符分割配置参数 ====================

# 异常值检测参数
IQR_MULTIPLIER = 1.5  # IQR异常值检测倍数

# 纵向连接字符检测参数
VERTICAL_CHAR_CONFIG = {
    'height_multiplier': 1.2,      # 高度至少是平均高度的倍数（进一步降低）
    'aspect_ratio_threshold': 1.5,  # 高宽比阈值（进一步降低）
    'min_width_ratio': 0.1,        # 最小宽度比例（进一步降低）
}

# 过宽字符检测参数
WIDE_CHAR_CONFIG = {
    'width_multiplier': 1.5,       # 宽度至少是平均宽度的倍数
    'min_height_ratio': 0.4,       # 最小高度比例（相对于平均高度）
    'max_width_height_ratio': 6.0,  # 最大宽高比
}

# 字符重分类参数
RECLASSIFY_CONFIG = {
    'narrow_threshold': 0.6,       # 窄字符宽度比例阈值
    'wide_threshold': 1.5,         # 宽字符宽度比例阈值
}

# 投影分析参数 - v2.0版本
PROJECTION_CONFIG = {
    # 基础分割验证参数
    'min_segment_height_ratio': 0.4,  # 最小分割高度比例
    'min_segment_width_ratio': 0.4,   # 最小分割宽度比例
    'min_final_height_ratio': 0.3,   # 最终最小高度比例
    
    # v2.0改进算法参数
    'gaussian_sigma_ratio': 20,       # 高斯核标准差比例 (sigma = mean_size / ratio)
    'adaptive_percentile_low': 25,    # 自适应阈值低分位数
    'adaptive_percentile_high': 75,   # 自适应阈值高分位数
    'adaptive_threshold_offset': 0.2, # 自适应阈值偏移系数
    'valley_min_distance_ratio': 4,   # 谷值最小间距比例 (distance = mean_size / ratio)
    'valley_prominence': 0.1,         # 谷值显著性阈值
    'boundary_exclusion_ratio': 0.1,  # 边界排除比例
    'validation_min_segment_ratio': 0.3, # 验证最小段长度比例
    'max_splits_limit': 5,            # 最大分割数限制
}

# 红红合并参数
RED_RED_MERGE_CONFIG = {
    'max_horizontal_distance': 1.5,    # 最大水平距离倍数（相对于参考宽度）
    'min_vertical_overlap_ratio': 0.3,  # 最小垂直重叠比例
    'width_ratio_range': (0.5, 2.5),   # 合并后宽度比例范围
    'height_ratio_range': (0.5, 3.0),  # 合并后高度比例范围
    'wide_char_multiplier': 2,          # 过宽字符判断倍数
    'wide_char_ratio': 0.3,            # 过宽字符比例
    'max_rounds': 10,                  # 最大合并轮数
}

# 红绿合并参数
RED_GREEN_MERGE_CONFIG = {
    'max_horizontal_distance': 2.0,    # 最大水平距离倍数（相对于参考宽度）
    'min_vertical_overlap_ratio': 0.3,  # 最小垂直重叠比例
    'width_ratio_range': (0.5, 3.5),   # 合并后宽度比例范围
    'height_ratio_range': (0.4, 3.5),  # 合并后高度比例范围
    'wide_char_multiplier': 2,          # 过宽字符判断倍数
    'wide_char_ratio': 0.3,            # 过宽字符比例
}

# 字符分类配置
CHAR_CLASSIFICATION_CONFIG = {
    'narrow_threshold_std_multiplier': 1.0,  # 窄字符判断的标准差倍数
    'binary_threshold': 128,  # 二值化阈值
    'dilation_kernel_size': 3,  # 膨胀操作核大小
    'min_char_size': 10,  # 最小字符尺寸阈值
    'dilation_iterations': 2,  # 膨胀操作迭代次数
}

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

# VL模型输出目录
VL_DIR = os.path.join(RESULTS_DIR, 'vl')