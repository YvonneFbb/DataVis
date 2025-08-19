"""
VL (Vision Language) 模块 - 使用视觉语言模型进行字符识别
主要支持SiliconFlow API，提供Qwen2.5-VL、GLM-4.5V等多种视觉模型
"""

from .core import VLRecognizer, process_characters_with_vl
from .providers import SiliconFlowVision

__all__ = [
    'VLRecognizer',
    'process_characters_with_vl',
    'SiliconFlowVision'
]