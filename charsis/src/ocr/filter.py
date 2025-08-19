"""
OCR过滤模块 - 使用ocrmac进行字符识别和过滤
"""
import os
import sys
import re
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OCR_FILTER_CONFIG

try:
    from ocrmac import ocrmac
    OCR_AVAILABLE = True
except ImportError:
    print("警告: ocrmac 未安装或不可用。OCR过滤功能将被禁用。")
    OCR_AVAILABLE = False


class OCRFilter:
    """
    OCR字符过滤器
    使用Apple Vision框架识别字符并过滤不准确的结果
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化OCR过滤器
        
        :param config: OCR配置参数，默认使用OCR_FILTER_CONFIG
        """
        self.config = config or OCR_FILTER_CONFIG.copy()
        self.enabled = self.config.get('enabled', True) and OCR_AVAILABLE
        
        if not self.enabled:
            print("OCR过滤已禁用")
            return
            
        # 初始化OCR引擎
        try:
            framework = self.config.get('framework', 'accurate')
            language = self.config.get('language_preference', ['zh-Hans'])
            
            # 根据框架类型创建OCR实例
            if framework == 'livetext':
                self.ocr_engine = lambda image: ocrmac.livetext_from_image(
                    image, language_preference=language
                ).recognize()
            else:
                self.ocr_engine = lambda image: ocrmac.OCR(
                    image, 
                    language_preference=language,
                    recognition_level=framework
                ).recognize()
                
            print(f"OCR引擎初始化成功: {framework} 模式, 语言: {language}")
            
        except Exception as e:
            print(f"OCR引擎初始化失败: {e}")
            self.enabled = False
    
    def is_chinese_char(self, text: str) -> bool:
        """
        检查文本是否包含中文字符
        
        :param text: 要检查的文本
        :return: 是否包含中文字符
        """
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_pattern.search(text))
    
    def validate_ocr_result(self, text: str, confidence: float) -> bool:
        """
        验证OCR识别结果是否有效
        
        :param text: 识别的文本
        :param confidence: 置信度
        :return: 是否有效
        """
        # 置信度检查
        if confidence < self.config['confidence_threshold']:
            return False
            
        # 空文本检查
        if not text.strip():
            return not self.config['allow_empty']
            
        # 文本长度检查
        if len(text) > self.config['max_text_length']:
            return False
            
        # 中文字符检查
        if self.config['chinese_only']:
            return self.is_chinese_char(text)
            
        return True
    
    def recognize_character(self, char_image: np.ndarray) -> Tuple[str, float, bool]:
        """
        识别单个字符图片
        
        :param char_image: 字符图片(numpy数组)
        :return: (识别文本, 置信度, 是否有效)
        """
        if not self.enabled:
            return "", 0.0, True  # OCR禁用时默认通过
            
        try:
            # 尺寸检查
            h, w = char_image.shape[:2]
            if w < self.config['min_char_width'] or h < self.config['min_char_height']:
                return "", 0.0, False
                
            # 转换为PIL图片
            if len(char_image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(char_image)
            
            # OCR识别
            results = self.ocr_engine(pil_image)
            
            if not results:
                return "", 0.0, not self.config['allow_empty']
            
            # 取置信度最高的结果
            best_result = max(results, key=lambda x: x[1])
            text, confidence, bbox = best_result
            
            # 验证结果
            is_valid = self.validate_ocr_result(text, confidence)
            
            return text.strip(), confidence, is_valid
            
        except Exception as e:
            print(f"OCR识别出错: {e}")
            return "", 0.0, True  # 出错时默认通过
    
    def filter_characters(self, original_image: np.ndarray, 
                         characters: List[Tuple]) -> Tuple[List[Tuple], Dict[str, Any]]:
        """
        批量过滤字符
        
        :param original_image: 原始图像
        :param characters: 字符列表 [(x, y, w, h, type), ...]
        :return: (过滤后的字符列表, 统计信息)
        """
        if not self.enabled:
            return characters, {'ocr_enabled': False}
            
        print(f"\n=== OCR过滤开始 ===")
        print(f"待处理字符数: {len(characters)}")
        
        # 转换为灰度图
        if len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = original_image
            
        valid_chars = []
        stats = {
            'total_chars': len(characters),
            'valid_chars': 0,
            'filtered_chars': 0,
            'low_confidence': 0,
            'empty_text': 0,
            'too_long': 0,
            'too_small': 0,
            'non_chinese': 0,
            'ocr_enabled': True
        }
        
        batch_size = self.config['batch_size']
        
        for i in range(0, len(characters), batch_size):
            batch = characters[i:i + batch_size]
            batch_end = min(i + batch_size, len(characters))
            
            print(f"处理批次: {i+1}-{batch_end}/{len(characters)}")
            
            for char in batch:
                x, y, w, h = char[:4]
                char_type = char[4] if len(char) > 4 else 'unknown'
                
                # 裁切字符图片
                char_img = gray_image[y:y+h, x:x+w]
                
                # OCR识别
                text, confidence, is_valid = self.recognize_character(char_img)
                
                if is_valid:
                    valid_chars.append(char)
                    stats['valid_chars'] += 1
                else:
                    stats['filtered_chars'] += 1
                    # 详细统计过滤原因
                    if confidence < self.config['confidence_threshold']:
                        stats['low_confidence'] += 1
                    elif not text.strip():
                        stats['empty_text'] += 1
                    elif len(text) > self.config['max_text_length']:
                        stats['too_long'] += 1
                    elif w < self.config['min_char_width'] or h < self.config['min_char_height']:
                        stats['too_small'] += 1
                    elif self.config['chinese_only'] and not self.is_chinese_char(text):
                        stats['non_chinese'] += 1
        
        print(f"OCR过滤完成:")
        print(f"  原字符数: {stats['total_chars']}")
        print(f"  有效字符: {stats['valid_chars']}")
        print(f"  过滤字符: {stats['filtered_chars']}")
        print(f"  过滤原因: 置信度低({stats['low_confidence']}), 空文本({stats['empty_text']}), "
              f"过长({stats['too_long']}), 过小({stats['too_small']}), 非中文({stats['non_chinese']})")
        
        return valid_chars, stats


def filter_characters_with_ocr(original_image: np.ndarray, 
                               characters: List[Tuple],
                               config: Dict[str, Any] = None) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    使用OCR过滤字符的便捷函数
    
    :param original_image: 原始图像
    :param characters: 字符列表
    :param config: OCR配置
    :return: (过滤后的字符列表, 统计信息)
    """
    filter_engine = OCRFilter(config)
    return filter_engine.filter_characters(original_image, characters)