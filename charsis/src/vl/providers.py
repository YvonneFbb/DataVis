"""
VL模型提供商实现 - 精简版，专注于文字识别
"""
import os
import sys
import base64
import json
import time
import requests
from typing import Dict, Any, Tuple
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from src.config import VL_CONFIG
except Exception as e:
    raise RuntimeError(
        f"无法导入 src.config（VL_CONFIG）。请从仓库根运行或设置 PYTHONPATH。原始错误: {e}"
    )


class SiliconFlowVision:
    """
    SiliconFlow Vision API客户端 - 精简版
    专注于文字识别，使用流式响应
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化SiliconFlow客户端
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
        """
        try:
            with Image.open(image_path) as img:
                # 转换为RGB模式（确保兼容性）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 压缩图片以减少传输时间
                max_size = (1920, 1920)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 编码为base64
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_data = buffer.getvalue()
                
                base64_string = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_string}"
                
        except Exception as e:
            raise Exception(f"图片编码失败: {e}")
    
    def recognize_text_stream(self, image_path: str, prompt: str, timeout: int = 120) -> Tuple[str, Dict[str, Any]]:
        """
        使用流式响应识别图片文字
        
        :param image_path: 图片路径
        :param prompt: 识别提示词
        :param timeout: 超时时间（秒）
        :return: (识别内容, 元数据)
        """
        print(f"\n=== 开始流式识别 ===")
        print(f"图片: {os.path.basename(image_path)}")
        print(f"超时时间: {timeout}秒")
        
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
                                    "detail": self.config.get('image_detail', 'high')
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": self.config.get('max_tokens', 2000),
                "temperature": self.config.get('temperature', 0.1),
                "stream": True  # 启用流式响应
            }
            
            # 发送流式请求
            print("发送流式请求...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=timeout
            )
            
            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                return "", {'error': error_msg}
            
            # 处理流式响应
            content = ""
            total_tokens = 0
            
            print(f"[{time.time() - start_time:.1f}s] 开始接收流式响应")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        
                        if data_str == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            
                            # 提取内容
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    chunk = choice['delta']['content']
                                    content += chunk
                                    
                                    # 显示进度
                                    if 'usage' in data:
                                        total_tokens = data['usage'].get('total_tokens', total_tokens)
                                        print(f"[{time.time() - start_time:.1f}s] Token使用: {total_tokens}")
                                        print(f"[{time.time() - start_time:.1f}s] +{len(chunk)}字符 (总计:{len(content)}字符)", end='\r')
                                        
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"\n[{response_time:.1f}s] ✅ 流式响应完成")
            print(f"\n📝 识别完成，共 {len(content)} 字符")
            
            # 处理编码问题
            try:
                if 'è' in content or 'ä' in content:
                    content = content.encode('latin-1').decode('utf-8')
            except:
                pass
            
            metadata = {
                'response_time': response_time,
                'content_length': len(content),
                'total_tokens': total_tokens,
                'model': self.model,
                'stream_mode': True
            }
            
            return content, metadata
            
        except requests.exceptions.Timeout:
            error_msg = f"请求超时（{timeout}秒）"
            return "", {'error': error_msg}
        except Exception as e:
            error_msg = f"流式识别失败: {e}"
            return "", {'error': error_msg}


# 兼容性别名
def SiliconFlowVision_old():
    """保持向后兼容的别名"""
    return SiliconFlowVision