"""Batch inference client for Qwen vision models."""
from __future__ import annotations
import json
import os
import time
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    base_url: str
    api_key_env: str = 'DASHSCOPE_API_KEY'
    completion_window: str = '24h'  # 最长等待时间: 24h-336h
    poll_interval: int = 60  # 轮询间隔（秒）
    max_requests_per_batch: int = 50000
    max_line_size_mb: float = 6.0  # 单行最大 6MB


class BatchClient:
    """批量推理客户端"""

    def __init__(self, cfg: BatchConfig):
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {cfg.api_key_env} 未设置")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key)

    def _encode_image(self, path: str) -> str:
        """Base64 编码图片"""
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _check_line_size(self, line: str) -> bool:
        """检查单行大小是否超限（6MB）"""
        size_mb = len(line.encode('utf-8')) / (1024 * 1024)
        if size_mb > self.cfg.max_line_size_mb:
            logger.warning(f"单行大小 {size_mb:.2f}MB 超过限制 {self.cfg.max_line_size_mb}MB")
            return False
        return True

    def generate_jsonl(self,
                       image_paths: List[str],
                       system_prompt: str,
                       model: str,
                       output_path: str,
                       enable_thinking: bool = False,
                       thinking_budget: int = 8192,
                       max_tokens: int = 512,
                       temperature: float = 0.0,
                       custom_ids: Optional[List[str]] = None) -> Tuple[int, int]:
        """
        生成批量推理的 JSONL 文件

        Args:
            custom_ids: 可选的自定义 ID 列表，长度需与 image_paths 一致

        Returns:
            (成功数量, 跳过数量)
        """
        if custom_ids and len(custom_ids) != len(image_paths):
            raise ValueError(f"custom_ids 数量 ({len(custom_ids)}) 与 image_paths ({len(image_paths)}) 不一致")

        logger.info(f"生成 JSONL 文件: {output_path}")
        logger.info(f"图片数量: {len(image_paths)}, 模型: {model}")

        success_count = 0
        skipped_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, image_path in enumerate(image_paths):
                try:
                    # 编码图片
                    image_b64 = self._encode_image(image_path)

                    # 构建请求体
                    body: Dict[str, Any] = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                                    {"type": "text", "text": "只输出上述 JSON"},
                                ],
                            },
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }

                    # qwen3-vl-plus 可以启用 thinking 模式（深度思考）
                    # 注意：批量推理不支持 stream 模式
                    if enable_thinking:
                        body["enable_thinking"] = True
                        body["thinking_budget"] = thinking_budget

                    # 使用自定义 ID 或默认文件名
                    custom_id = custom_ids[idx] if custom_ids else os.path.basename(image_path)

                    # 构建请求行
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body
                    }

                    # 序列化并检查大小
                    line = json.dumps(request, separators=(',', ':'), ensure_ascii=False) + '\n'

                    if not self._check_line_size(line):
                        logger.warning(f"跳过 {image_path}（单行过大）")
                        skipped_count += 1
                        continue

                    f.write(line)
                    success_count += 1

                    if (idx + 1) % 100 == 0:
                        logger.info(f"已处理 {idx + 1}/{len(image_paths)} 张图片")

                except Exception as e:
                    logger.error(f"处理 {image_path} 失败: {e}")
                    skipped_count += 1

        logger.info(f"JSONL 生成完成: 成功 {success_count}, 跳过 {skipped_count}")
        return success_count, skipped_count

    def upload_file(self, jsonl_path: str) -> str:
        """
        上传 JSONL 文件

        Returns:
            file_id
        """
        logger.info(f"上传文件: {jsonl_path}")
        file_object = self.client.files.create(
            file=Path(jsonl_path),
            purpose="batch"
        )
        logger.info(f"文件上传成功, file_id: {file_object.id}")
        return file_object.id

    def create_batch(self, file_id: str, task_name: str = "", task_desc: str = "") -> str:
        """
        创建批量推理任务

        Returns:
            batch_id
        """
        logger.info(f"创建批量任务: file_id={file_id}")

        metadata = {}
        if task_name:
            metadata['ds_name'] = task_name
        if task_desc:
            metadata['ds_description'] = task_desc

        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=self.cfg.completion_window,
            metadata=metadata if metadata else None
        )
        logger.info(f"批量任务创建成功, batch_id: {batch.id}, status: {batch.status}")
        return batch.id

    def get_status(self, batch_id: str) -> Dict[str, Any]:
        """查询批量任务状态"""
        batch = self.client.batches.retrieve(batch_id)
        return {
            'id': batch.id,
            'status': batch.status,
            'request_counts': {
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed,
            },
            'output_file_id': batch.output_file_id,
            'error_file_id': batch.error_file_id,
            'created_at': batch.created_at,
            'completed_at': batch.completed_at,
        }

    def wait_for_completion(self, batch_id: str) -> Dict[str, Any]:
        """
        等待批量任务完成（轮询）

        Returns:
            最终状态信息
        """
        logger.info(f"等待批量任务完成: {batch_id}")
        logger.info(f"轮询间隔: {self.cfg.poll_interval}秒")

        while True:
            status_info = self.get_status(batch_id)
            status = status_info['status']
            counts = status_info['request_counts']

            logger.info(f"状态: {status}, 进度: {counts['completed']}/{counts['total']} "
                       f"(失败: {counts['failed']})")

            if status in ['completed', 'failed', 'expired', 'cancelled']:
                logger.info(f"任务结束: {status}")
                return status_info

            time.sleep(self.cfg.poll_interval)

    def download_results(self, file_id: str, output_path: str) -> None:
        """下载结果文件"""
        logger.info(f"下载结果: file_id={file_id}, output={output_path}")
        content = self.client.files.content(file_id)
        content.write_to_file(output_path)
        logger.info(f"结果已保存: {output_path}")

    def parse_results(self, result_jsonl: str) -> List[Dict[str, Any]]:
        """
        解析批量推理结果

        Returns:
            [{'custom_id': ..., 'content': ..., 'error': ...}, ...]
        """
        results = []
        with open(result_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    custom_id = item.get('custom_id', '')

                    # 检查是否有错误
                    error = item.get('error')
                    if error:
                        results.append({
                            'custom_id': custom_id,
                            'content': None,
                            'error': error
                        })
                        continue

                    # 提取响应内容
                    response = item.get('response', {})
                    body = response.get('body', {})
                    choices = body.get('choices', [])

                    if choices:
                        message = choices[0].get('message', {})
                        content = message.get('content', '')
                        reasoning = message.get('reasoning_content', '')

                        results.append({
                            'custom_id': custom_id,
                            'content': content,
                            'reasoning': reasoning,
                            'error': None,
                            'usage': body.get('usage', {}),
                        })
                    else:
                        results.append({
                            'custom_id': custom_id,
                            'content': None,
                            'error': 'No choices in response'
                        })

                except Exception as e:
                    logger.error(f"解析结果行失败: {e}")

        logger.info(f"解析结果完成: {len(results)} 条")
        return results

    def save_batch_state(self, work_dir: str, batch_id: str, status: str = 'submitted') -> None:
        """保存批量任务状态（用于断点恢复）"""
        state_file = os.path.join(work_dir, 'batch_state.json')
        state = {
            'batch_id': batch_id,
            'status': status,
            'timestamp': time.time(),
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"批量任务状态已保存: {state_file}")

    def load_batch_state(self, work_dir: str) -> Optional[Dict[str, Any]]:
        """加载批量任务状态"""
        state_file = os.path.join(work_dir, 'batch_state.json')
        if not os.path.exists(state_file):
            return None
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载批量任务状态失败: {e}")
            return None

    def resume_batch_inference(self, work_dir: str) -> Dict[str, Any]:
        """从断点恢复批量推理任务"""
        state = self.load_batch_state(work_dir)
        if not state:
            raise RuntimeError("未找到批量任务状态文件")

        batch_id = state['batch_id']
        logger.info(f"恢复批量任务: {batch_id}")

        # 等待完成
        status_info = self.wait_for_completion(batch_id)

        # 下载结果
        result_jsonl = os.path.join(work_dir, 'batch_results.jsonl')
        if status_info['output_file_id']:
            self.download_results(status_info['output_file_id'], result_jsonl)

        # 下载错误文件（如果有）
        if status_info['error_file_id']:
            error_jsonl = os.path.join(work_dir, 'batch_errors.jsonl')
            self.download_results(status_info['error_file_id'], error_jsonl)

        # 解析结果
        results = self.parse_results(result_jsonl) if status_info['output_file_id'] else []

        # 更新状态
        self.save_batch_state(work_dir, batch_id, status=status_info['status'])

        return {
            'batch_id': batch_id,
            'status': status_info['status'],
            'results': results,
            'result_file': result_jsonl,
            'request_counts': status_info['request_counts'],
            'resumed': True,
        }

    def run_batch_inference(self,
                           image_paths: List[str],
                           system_prompt: str,
                           model: str,
                           work_dir: str,
                           task_name: str = "",
                           enable_thinking: bool = False,
                           thinking_budget: int = 8192,
                           max_tokens: int = 512,
                           temperature: float = 0.0,
                           resume: bool = True) -> Dict[str, Any]:
        """
        一键执行完整批量推理流程

        Args:
            resume: 是否尝试从断点恢复

        Returns:
            {
                'batch_id': ...,
                'status': ...,
                'results': [...],
                'result_file': ...,
                'resumed': bool,  # 是否从断点恢复
            }
        """
        os.makedirs(work_dir, exist_ok=True)

        # 尝试从断点恢复
        if resume:
            state = self.load_batch_state(work_dir)
            if state:
                logger.info(f"检测到未完成的批量任务: {state['batch_id']}")
                try:
                    return self.resume_batch_inference(work_dir)
                except Exception as e:
                    logger.warning(f"恢复任务失败: {e}，将创建新任务")

        # 1. 生成 JSONL
        jsonl_path = os.path.join(work_dir, 'batch_input.jsonl')
        success_count, skipped_count = self.generate_jsonl(
            image_paths=image_paths,
            system_prompt=system_prompt,
            model=model,
            output_path=jsonl_path,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if success_count == 0:
            raise RuntimeError("没有成功生成任何请求")

        # 2. 上传文件
        file_id = self.upload_file(jsonl_path)

        # 3. 创建任务
        batch_id = self.create_batch(file_id, task_name=task_name)

        # 保存状态（用于断点恢复）
        self.save_batch_state(work_dir, batch_id, status='submitted')

        # 4. 等待完成
        status_info = self.wait_for_completion(batch_id)

        # 5. 下载结果
        result_jsonl = os.path.join(work_dir, 'batch_results.jsonl')
        if status_info['output_file_id']:
            self.download_results(status_info['output_file_id'], result_jsonl)

        # 6. 下载错误文件（如果有）
        if status_info['error_file_id']:
            error_jsonl = os.path.join(work_dir, 'batch_errors.jsonl')
            self.download_results(status_info['error_file_id'], error_jsonl)

        # 7. 解析结果
        results = self.parse_results(result_jsonl) if status_info['output_file_id'] else []

        # 更新状态
        self.save_batch_state(work_dir, batch_id, status=status_info['status'])

        return {
            'batch_id': batch_id,
            'status': status_info['status'],
            'results': results,
            'result_file': result_jsonl,
            'request_counts': status_info['request_counts'],
            'resumed': False,
        }
