"""
VL (Vision Language) 视觉语言模型模块

专注于古籍文字识别，使用流式响应提升性能
支持SiliconFlow API，提供Qwen2.5-VL等视觉模型
"""

from .providers import SiliconFlowVision
from .text_recognition import VLTextRecognizer, recognize_image

__all__ = [
    'SiliconFlowVision',
    'VLTextRecognizer', 
    'recognize_image'
]