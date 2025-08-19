"""
VL模型提供商实现 - SiliconFlow API集成
"""
import os
import sys
import base64
import json
import time
import requests
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
from PIL import Image

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VL_CONFIG


class SiliconFlowVision:
    """
    SiliconFlow Vision API客户端
    支持Qwen2.5-VL等多种视觉语言模型
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化SiliconFlow客户端
        
        :param config: VL配置，默认使用VL_CONFIG
        """
        self.config = config or VL_CONFIG.copy()
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']
        self.model = self.config['model']
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        print(f"SiliconFlow VL客户端初始化:")
        print(f"  模型: {self.model}")
        print(f"  API地址: {self.base_url}")
        print(f"  超时时间: {self.config['timeout']}秒")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图片编码为base64格式
        
        :param image_path: 图片路径
        :return: base64编码的图片
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
            # 检测图片格式
            image_format = Image.open(image_path).format.lower()
            if image_format == 'jpeg':
                image_format = 'jpg'
                
            return f"data:image/{image_format};base64,{base64_image}"
            
        except Exception as e:
            raise Exception(f"图片编码失败: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试API连接和认证
        
        :return: 测试结果
        """
        print(f"\n=== SiliconFlow API连通性测试 ===")
        
        # 创建一个简单的测试图片(纯文本)
        test_image = Image.new('RGB', (100, 100), color='white')
        test_image_path = '/tmp/test_connection.png'
        test_image.save(test_image_path)
        
        try:
            # 编码测试图片
            base64_image = self.encode_image_to_base64(test_image_path)
            
            # 构建请求
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": self.config['image_detail']
                                }
                            },
                            {
                                "type": "text",
                                "text": "这是一张测试图片，请简单描述一下你看到了什么。"
                            }
                        ]
                    }
                ],
                "max_tokens": self.config['max_tokens'],
                "temperature": self.config['temperature']
            }
            
            print(f"发送测试请求到: {self.base_url}/chat/completions")
            print(f"使用模型: {self.model}")
            
            # 发送请求
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config['timeout']
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # 清理测试文件
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                print(f"✅ 连接成功!")
                print(f"响应时间: {response_time:.2f}秒")
                print(f"模型回复: {content[:100]}...")
                
                return {
                    'success': True,
                    'response_time': response_time,
                    'model': self.model,
                    'content': content,
                    'usage': result.get('usage', {})
                }
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': response.status_code
                }
                
        except requests.RequestException as e:
            error_msg = f"网络请求错误: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"测试连接时出错: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def recognize_character(self, image_path: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        识别单个字符图片
        
        :param image_path: 字符图片路径
        :return: (识别文本, 置信度估计, 详细信息)
        """
        try:
            # 编码图片
            base64_image = self.encode_image_to_base64(image_path)
            
            # 构建请求
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": self.config['image_detail']
                                }
                            },
                            {
                                "type": "text",
                                "text": self.config['prompt_template']
                            }
                        ]
                    }
                ],
                "max_tokens": self.config['max_tokens'],
                "temperature": self.config['temperature']
            }
            
            # 发送请求
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config['timeout']
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                
                # 简单的置信度估计（基于响应特征）
                confidence = self._estimate_confidence(content, result)
                
                # 处理识别结果
                if content in ['无法识别', '不清楚', '看不清', '']:
                    content = ""
                    confidence = 0.0
                
                detail_info = {
                    'response_time': end_time - start_time,
                    'usage': result.get('usage', {}),
                    'model': self.model,
                    'raw_response': content
                }
                
                return content, confidence, detail_info
                
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                return "", 0.0, {'error': error_msg}
                
        except Exception as e:
            error_msg = f"字符识别失败: {e}"
            return "", 0.0, {'error': error_msg}
    
    def _estimate_confidence(self, content: str, api_response: Dict[str, Any]) -> float:
        """
        估计识别置信度
        
        :param content: 识别内容
        :param api_response: API完整响应
        :return: 置信度估计 (0.0-1.0)
        """
        if not content or content in ['无法识别', '不清楚', '看不清']:
            return 0.0
        
        # 基础置信度
        base_confidence = 0.7
        
        # 根据内容特征调整
        if len(content) == 1 and '\u4e00' <= content <= '\u9fff':  # 单个中文字符
            base_confidence += 0.2
        elif len(content) > 1:  # 多字符可能是误识别
            base_confidence -= 0.3
        
        # 根据token使用量调整（更多token可能表示更复杂的推理）
        usage = api_response.get('usage', {})
        completion_tokens = usage.get('completion_tokens', 0)
        if completion_tokens > 10:  # 如果输出很长，可能不是单字符
            base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))


def test_siliconflow_connection(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    测试SiliconFlow连接的便捷函数
    
    :param config: VL配置
    :return: 测试结果
    """
    client = SiliconFlowVision(config)
    return client.test_connection()


if __name__ == '__main__':
    # 运行连接测试
    result = test_siliconflow_connection()
    
    if result['success']:
        print(f"\n🎉 SiliconFlow API连接测试成功!")
        print(f"模型: {result['model']}")
        print(f"响应时间: {result['response_time']:.2f}秒")
        if 'usage' in result:
            usage = result['usage']
            print(f"Token使用: {usage.get('total_tokens', 0)} (输入: {usage.get('prompt_tokens', 0)}, 输出: {usage.get('completion_tokens', 0)})")
    else:
        print(f"\n❌ 连接测试失败: {result.get('error', '未知错误')}")
        exit(1)