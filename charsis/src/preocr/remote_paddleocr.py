#!/usr/bin/env python3
"""
远程PaddleOCR客户端（preOCR 阶段：文本区域检测/字符识别服务）
"""
import os
import sys
import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 确保能从仓库根导入 src.config
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from src.config import SEGMENTS_DIR, PREOCR_DIR, OCR_REMOTE_CONFIG
except Exception as e:
    raise RuntimeError(
        f"无法导入 src.config（SEGMENTS_DIR/PREOCR_DIR/OCR_REMOTE_CONFIG）。请从仓库根运行或设置 PYTHONPATH。原始错误: {e}"
    )
from src.utils.path import ensure_dir


class RemotePaddleOCRClient:
    """远程PaddleOCR服务客户端"""

    def __init__(self, server_url: str | None = None, timeout: int | None = None):
        self.server_url = server_url or OCR_REMOTE_CONFIG.get('server_url', 'http://127.0.0.1:8000')
        self.timeout = timeout or OCR_REMOTE_CONFIG.get('timeout', 30)
        self.session = requests.Session()
        print(f"正在连接到PaddleOCR服务: {self.server_url}")
        self._check_connection()

    def _check_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"PaddleOCR服务连接成功")
                print(f"   状态: {health.get('status', 'unknown')}")
                print(f"   GPU可用: {health.get('gpu_available', False)}")
                return True
            else:
                print(f"⚠️  PaddleOCR服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ PaddleOCR服务连接失败: {e}")
            return False

    def get_service_info(self) -> Optional[Dict]:
        try:
            response = self.session.get(f"{self.server_url}/", timeout=5)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"获取服务信息失败: {e}")
            return None

    def predict_single_character(self, image_path: Path) -> Dict[str, Any]:
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
            return self._error_result(f'请求超时 (>{self.timeout}s)')
        except Exception as e:
            return self._error_result(f'异常: {str(e)}')

    def predict_full_document(self, image_path: Path) -> Dict[str, Any]:
        try:
            if not image_path.exists():
                return self._error_result(f"图片不存在: {image_path}")
            start_time = time.time()
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, self._get_mime_type(image_path))}
                response = self.session.post(
                    f"{self.server_url}/ocr/predict",
                    files=files,
                    timeout=60
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
            return self._error_result(f'异常: {str(e)}')

    def _get_mime_type(self, image_path: Path) -> str:
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
        return {'text': '', 'confidence': 0.0, 'process_time': process_time, 'success': False, 'message': message}

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()


if __name__ == "__main__":
    # 简单连通性测试
    client = RemotePaddleOCRClient()
    print('Service info:', client.get_service_info())