#!/usr/bin/env python3
"""
远程PaddleOCR客户端
连接到Windows主机上的PaddleOCR服务进行OCR识别
"""

import os
import sys
import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEGMENTS_DIR, OCR_DIR, OCR_REMOTE_CONFIG
from utils.path import ensure_dir

class RemotePaddleOCRClient:
    """远程PaddleOCR服务客户端"""
    
    def __init__(self, server_url: str | None = None, timeout: int | None = None):
        # 支持通过入参覆盖，其次读取配置/环境变量，最后退回默认
        self.server_url = server_url or OCR_REMOTE_CONFIG.get('server_url', 'http://127.0.0.1:8000')
        self.timeout = timeout or OCR_REMOTE_CONFIG.get('timeout', 30)
        self.session = requests.Session()
        
        # 连接检查
        print(f"正在连接到PaddleOCR服务: {self.server_url}")
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """初始化时检查连接状态"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ PaddleOCR服务连接成功")
                print(f"   状态: {health.get('status', 'unknown')}")
                print(f"   GPU可用: {health.get('gpu_available', False)}")
                return True
            else:
                print(f"⚠️  PaddleOCR服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ PaddleOCR服务连接失败: {e}")
            print("   请确保:")
            print("   1. 主机PaddleOCR服务正在运行")
            print("   2. 网络连通性正常")
            print("   3. 防火墙允许访问8000端口")
            return False
    
    def get_service_info(self) -> Optional[Dict]:
        """获取服务信息"""
        try:
            response = self.session.get(f"{self.server_url}/", timeout=5)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取服务信息失败: {e}")
            return None
    
    def get_model_info(self) -> Optional[Dict]:
        """获取模型信息"""
        try:
            response = self.session.get(f"{self.server_url}/models/info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取模型信息失败: {e}")
            return None
    
    def predict_single_character(self, image_path: Path) -> Dict[str, Any]:
        """
        识别单个字符图片
        
        Args:
            image_path: 字符图片路径
            
        Returns:
            dict: 包含text、confidence等信息的结果
        """
        try:
            if not image_path.exists():
                return self._error_result(f"图片不存在: {image_path}")
            
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, self._get_mime_type(image_path))}
                response = self.session.post(
                    f"{self.server_url}/ocr/predict",
                    files=files,
                    timeout=self.timeout
                )
            
            process_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success') and result.get('text_regions'):
                    # 对于单个字符，取第一个识别结果
                    first_region = result['text_regions'][0]
                    return {
                        'text': first_region.get('text', '').strip(),
                        'confidence': first_region.get('confidence', 0.0),
                        'process_time': process_time,
                        'success': True,
                        'total_regions': len(result['text_regions']),
                        'server_time': result.get('total_time', 0.0)
                    }
                else:
                    return {
                        'text': '',
                        'confidence': 0.0,
                        'process_time': process_time,
                        'success': False,
                        'message': result.get('message', '无识别结果')
                    }
            else:
                return self._error_result(f'HTTP {response.status_code}: {response.text}', process_time)
                
        except requests.exceptions.Timeout:
            return self._error_result(f'请求超时 (>{self.timeout}s)', time.time() - start_time if 'start_time' in locals() else 0)
        except Exception as e:
            return self._error_result(f'异常: {str(e)}', time.time() - start_time if 'start_time' in locals() else 0)
    
    def predict_batch_characters(self, image_paths: List[Path], max_batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        批量识别字符图片
        
        Args:
            image_paths: 字符图片路径列表
            max_batch_size: 单次批量请求的最大图片数
            
        Returns:
            List[Dict]: 识别结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(image_paths), max_batch_size):
            batch_paths = image_paths[i:i + max_batch_size]
            print(f"处理批次 {i//max_batch_size + 1}: {len(batch_paths)} 张图片")
            
            # 对当前批次的每个图片进行识别
            for image_path in batch_paths:
                result = self.predict_single_character(image_path)
                results.append({
                    'filename': image_path.name,
                    **result
                })
        
        return results
    
    def predict_full_document(self, image_path: Path) -> Dict[str, Any]:
        """
        识别完整文档图片
        
        Args:
            image_path: 文档图片路径
            
        Returns:
            dict: 包含所有文本区域的识别结果
        """
        try:
            if not image_path.exists():
                return self._error_result(f"图片不存在: {image_path}")
            
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, self._get_mime_type(image_path))}
                response = self.session.post(
                    f"{self.server_url}/ocr/predict",
                    files=files,
                    timeout=60  # 文档识别使用更长超时
                )
            
            process_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    return {
                        'text_regions': result.get('text_regions', []),
                        'total_regions': len(result.get('text_regions', [])),
                        'process_time': process_time,
                        'server_time': result.get('total_time', 0.0),
                        'success': True
                    }
                else:
                    return self._error_result(result.get('message', '识别失败'), process_time)
            else:
                return self._error_result(f'HTTP {response.status_code}', process_time)
                
        except Exception as e:
            return self._error_result(f'异常: {str(e)}', time.time() - start_time if 'start_time' in locals() else 0)
    
    def _get_mime_type(self, image_path: Path) -> str:
        """根据文件扩展名返回MIME类型"""
        ext = image_path.suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    def _error_result(self, message: str, process_time: float = 0.0) -> Dict[str, Any]:
        """生成错误结果"""
        return {
            'text': '',
            'confidence': 0.0,
            'process_time': process_time,
            'success': False,
            'message': message
        }
    
    def __del__(self):
        """析构函数，关闭session"""
        if hasattr(self, 'session'):
            self.session.close()


def test_remote_paddleocr():
    """测试远程PaddleOCR客户端"""
    print("=== 远程PaddleOCR客户端测试 ===")
    
    # 初始化客户端
    client = RemotePaddleOCRClient()
    
    # 1. 服务信息测试
    print("\n1. 获取服务信息...")
    service_info = client.get_service_info()
    if service_info:
        print(f"   服务: {service_info.get('service', 'Unknown')}")
        print(f"   版本: {service_info.get('version', 'Unknown')}")
    
    # 2. 模型信息测试
    print("\n2. 获取模型信息...")
    model_info = client.get_model_info()
    if model_info:
        print(f"   模型: {model_info.get('model_name', 'Unknown')}")
        print(f"   GPU加速: {model_info.get('gpu_enabled', False)}")
    
    # 3. 单字符识别测试
    project_root = Path(__file__).parent.parent.parent
    
    # 测试已分割的字符
    char_dir = project_root / 'data' / 'results' / 'ocr' / 'demo_preprocessed_alpha1.5_beta10'
    if char_dir.exists():
        char_images = [f for f in char_dir.glob('*.png') 
                      if not f.name.startswith('empty_') and f.name not in ['annotated.png', 'ocr_results.json']]
        
        if char_images:
            test_image = char_images[0]  # 测试第一张图片
            print(f"\n3. 单字符识别测试: {test_image.name}")
            result = client.predict_single_character(test_image)
            
            if result['success']:
                print(f"   ✅ 识别成功: '{result['text']}'")
                print(f"   置信度: {result['confidence']:.3f}")
                print(f"   处理耗时: {result['process_time']:.3f}秒")
                print(f"   服务端耗时: {result.get('server_time', 0):.3f}秒")
            else:
                print(f"   ❌ 识别失败: {result['message']}")
    
    # 4. 完整文档识别测试
    demo_image = project_root / 'data' / 'raw' / 'demo.jpg'
    if demo_image.exists():
        print(f"\n4. 完整文档识别测试: {demo_image.name}")
        result = client.predict_full_document(demo_image)
        
        if result['success']:
            print(f"   ✅ 识别成功: 找到 {result['total_regions']} 个文本区域")
            print(f"   处理耗时: {result['process_time']:.3f}秒")
            
            # 显示前10个识别结果
            for i, region in enumerate(result['text_regions'][:10]):
                text = region.get('text', '')
                confidence = region.get('confidence', 0)
                print(f"      {i+1:2d}. '{text}' (置信度: {confidence:.3f})")
            
            if result['total_regions'] > 10:
                print(f"      ... 还有 {result['total_regions']-10} 个结果")
        else:
            print(f"   ❌ 识别失败: {result['message']}")
    
    print(f"\n=== 测试完成 ===")


if __name__ == "__main__":
    test_remote_paddleocr()