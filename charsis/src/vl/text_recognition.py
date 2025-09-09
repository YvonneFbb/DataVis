"""
VL模型文字识别模块 - 专注于古籍文字识别，不包含位置检测
"""
import os
import sys
import json
from typing import Dict, Any, List

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VL_CONFIG, VL_DIR
from src.utils.path import ensure_dir
from vl.providers import SiliconFlowVision


class VLTextRecognizer:
    """
    VL模型文字识别器
    专注于古籍文字识别，使用流式响应
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化VL文字识别器
        """
        self.config = config or VL_CONFIG.copy()
        
        # 从环境变量获取API key
        api_key = os.environ.get('SILICONFLOW_API_KEY', '')
        if api_key:
            self.config['api_key'] = api_key
        
        # 设置合理的超时时间
        self.config['timeout'] = 120  # 2分钟超时
        
        self.client = SiliconFlowVision(self.config)
        
        # 简洁的识别提示词
        self.recognition_prompt = """请识别这张古籍图片中的所有汉字。

要求：
1. 按照从右到左、从上到下的阅读顺序
2. 只输出识别到的汉字，用空格分隔
3. 如果有明显的列分隔，用 | 分隔不同列
4. 不要添加任何解释或标注

示例输出：
識 貧 賤 者 不 足 與 言 | 仁 義 禮 智 信"""
        
        print(f"VL文字识别器初始化完成")
        print(f"模型: {self.config['model']}")
    
    def recognize_text(self, image_path: str) -> Dict[str, Any]:
        """
        识别图片中的文字
        
        :param image_path: 图片路径
        :return: 识别结果
        """
        if not os.path.exists(image_path):
            return {'error': f'图片文件不存在: {image_path}'}
        
        print(f"\n=== 开始文字识别 ===")
        print(f"图片: {os.path.basename(image_path)}")
        
        try:
            # 使用流式识别
            content, metadata = self.client.recognize_text_stream(
                image_path, 
                self.recognition_prompt, 
                self.config['timeout']
            )
            
            if 'error' in metadata:
                return {'error': metadata['error']}
            
            # 分析识别结果
            analysis = self._analyze_text(content)
            
            result = {
                'success': True,
                'image_path': image_path,
                'recognized_text': content.strip(),
                'analysis': analysis,
                'metadata': metadata
            }
            
            print(f"✅ 识别成功，共识别 {analysis['character_count']} 个汉字")
            print(f"响应时间: {metadata.get('response_time', 0):.1f}秒")
            
            return result
            
        except Exception as e:
            error_msg = f"文字识别失败: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def _analyze_text(self, content: str) -> Dict[str, Any]:
        """
        分析识别结果
        """
        import re
        
        # 提取汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', content)
        
        # 分析列结构
        columns = content.split('|') if '|' in content else [content]
        column_count = len(columns)
        
        analysis = {
            'content_length': len(content),
            'character_count': len(chinese_chars),
            'unique_characters': len(set(chinese_chars)),
            'column_count': column_count,
            'first_20_chars': ''.join(chinese_chars[:20]),
            'has_column_separators': '|' in content
        }
        
        return analysis
    
    def save_result(self, result: Dict[str, Any], output_path: str = None) -> str:
        """
        保存识别结果
        """
        if output_path is None:
            ensure_dir(VL_DIR)
            image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
            output_path = os.path.join(VL_DIR, f"{image_name}_recognition_result.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"📝 识别结果已保存: {output_path}")
        return output_path


def recognize_image(image_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    便捷函数：识别单张图片的文字
    
    :param image_path: 图片路径
    :param output_path: 结果保存路径（可选）
    :return: 识别结果
    """
    recognizer = VLTextRecognizer()
    result = recognizer.recognize_text(image_path)
    
    if result.get('success'):
        recognizer.save_result(result, output_path)
    
    return result


if __name__ == '__main__':
    # 示例用法
    from config import PREPROCESSED_DIR
    
    test_image = os.path.join(PREPROCESSED_DIR, "demo_preprocessed_alpha1.5_beta10.jpg")
    
    if os.path.exists(test_image):
        print("=== VL文字识别测试 ===")
        result = recognize_image(test_image)
        
        if result.get('success'):
            analysis = result['analysis']
            print(f"\n🎉 识别成功!")
            print(f"📝 识别文字: {result['recognized_text'][:100]}...")
            print(f"📊 统计信息:")
            print(f"   - 汉字数量: {analysis['character_count']}")
            print(f"   - 唯一字符: {analysis['unique_characters']}")
            print(f"   - 列数: {analysis['column_count']}")
            print(f"   - 前20字符: {analysis['first_20_chars']}")
        else:
            print(f"❌ 识别失败: {result.get('error')}")
    else:
        print(f"测试图片不存在: {test_image}")