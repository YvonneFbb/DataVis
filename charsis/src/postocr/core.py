"""Post-OCR filtering using Doubao vision model."""
from __future__ import annotations

import json
import os
import re
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src.config import POSTOCR_CONFIG, SEGMENTS_DIR, POSTOCR_DIR
except Exception as e:
    raise RuntimeError(
        f"无法导入必要配置 (POSTOCR_CONFIG/SEGMENTS_DIR/POSTOCR_DIR)。原始错误: {e}"
    )

from src.utils.path import ensure_dir
from src.postocr.vision_client import VisionClient, VisionConfig
from src.postocr.batch_client import BatchClient, BatchConfig

logger = logging.getLogger(__name__)


def _safe_filename(text: str, max_length: int = 20) -> str:
    text = (text or '').strip()
    if not text:
        return 'unknown'
    safe = re.sub(r'[<>:"/\\|?*]', '_', text)
    return safe[:max_length]


def _load_existing_results(result_path: str) -> Dict[str, Any]:
    """加载已有的处理结果（如果存在）

    Returns:
        {'results_map': {original_file: entry}, 'batch_ids': []}
    """
    if not os.path.exists(result_path):
        return {'results_map': {}, 'batch_ids': []}

    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 建立文件名到结果的映射
        results_map = {}
        for entry in data.get('results', []):
            original_file = entry.get('original_file')
            if original_file:
                results_map[original_file] = entry

        # 保留历史batch_ids
        batch_ids = data.get('batch_ids', [])
        if isinstance(batch_ids, str):  # 兼容旧格式 batch_id: "xxx"
            batch_ids = [batch_ids]
        elif not batch_ids and 'batch_id' in data:
            batch_ids = [data['batch_id']]

        return {'results_map': results_map, 'batch_ids': batch_ids}
    except Exception as e:
        logger.warning(f"读取已有结果失败: {e}")
        return {'results_map': {}, 'batch_ids': []}


def _save_incremental_results(result_path: str, new_entries: List[Dict[str, Any]],
                               batch_id: str) -> Dict[str, Any]:
    """增量保存处理结果（字符级粒度）

    Args:
        result_path: postocr_results.json 路径
        new_entries: 新处理的结果列表
        batch_id: 当前批量任务ID

    Returns:
        统计信息
    """
    # 1. 加载已有结果
    existing = _load_existing_results(result_path)
    results_map = existing['results_map']
    batch_ids = existing['batch_ids']

    # 2. 更新/添加新结果（按 original_file 覆盖）
    for entry in new_entries:
        original_file = entry.get('original_file')
        if original_file:
            # 保留时间戳
            if original_file not in results_map:
                entry['first_processed_at'] = time.time()
            else:
                entry['first_processed_at'] = results_map[original_file].get('first_processed_at', time.time())
            entry['last_updated_at'] = time.time()

            results_map[original_file] = entry

    # 3. 添加当前batch_id
    if batch_id and batch_id not in batch_ids:
        batch_ids.append(batch_id)

    # 4. 重新计算统计信息
    all_entries = list(results_map.values())
    stats = {
        'total_files': len(all_entries),
        'accepted': sum(1 for e in all_entries if e.get('status') == 'accepted'),
        'filtered_out': sum(1 for e in all_entries if e.get('status') == 'filtered_out'),
        'error': sum(1 for e in all_entries if e.get('status') == 'error'),
    }

    # 5. 保存合并后的结果
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'results': all_entries,
            'batch_ids': batch_ids,
            'last_updated': time.time()
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"增量保存结果: {result_path}, 总计 {len(all_entries)} 个字符")
    return stats


def _parse_ark_response(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[-1]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
    try:
        data = json.loads(cleaned)
    except Exception:
        logger.warning("Ark 响应无法解析为 JSON: %s", raw[:200])
        return {
            'raw': raw,
            'parsed': None,
            'single_character': None,
            'recognizable': None,
            'has_noise': None,
            'character': None,
            'comment': raw,
        }
    return {
        'raw': raw,
        'parsed': data,
        'single_character': bool(data.get('single_character')),
        'recognizable': bool(data.get('recognizable')),
        'has_noise': bool(data.get('has_noise')),
        'character': (data.get('character') or '').strip(),
        'comment': data.get('comment'),
    }


def _build_vision_config() -> Optional[VisionConfig]:
    """根据配置构建视觉模型客户端配置"""
    if not POSTOCR_CONFIG.get('enabled', False):
        return None

    provider = str(POSTOCR_CONFIG.get('provider', 'doubao')).lower()
    provider_cfg = POSTOCR_CONFIG.get(provider, {})

    if not provider_cfg:
        logger.warning(f"未找到 provider '{provider}' 的配置，PostOCR 将被禁用")
        return None

    model = str(provider_cfg.get('model', ''))
    enable_thinking_cfg = provider_cfg.get('enable_thinking', 'auto')

    # 处理 enable_thinking 配置
    if enable_thinking_cfg == 'auto':
        enable_thinking = 'qwen3-vl-plus' in model.lower()
    else:
        enable_thinking = bool(enable_thinking_cfg)

    return VisionConfig(
        base_url=str(provider_cfg.get('base_url', '')),
        model=model,
        api_key_env=str(provider_cfg.get('api_key_env', 'ARK_API_KEY')),
        timeout=int(provider_cfg.get('timeout', 60)),
        temperature=float(provider_cfg.get('temperature', 0.0)),
        max_tokens=int(provider_cfg.get('max_tokens', 256)),
        enable_thinking=enable_thinking,
        thinking_budget=int(provider_cfg.get('thinking_budget', 8192)),
    )


def _build_batch_config() -> Optional[BatchConfig]:
    """根据配置构建批量推理客户端配置"""
    if not POSTOCR_CONFIG.get('enabled', False):
        return None

    provider = str(POSTOCR_CONFIG.get('provider', 'doubao')).lower()
    provider_cfg = POSTOCR_CONFIG.get(provider, {})

    if not provider_cfg:
        logger.warning(f"未找到 provider '{provider}' 的配置，PostOCR 将被禁用")
        return None

    batch_cfg = POSTOCR_CONFIG.get('batch', {})

    return BatchConfig(
        base_url=str(provider_cfg.get('base_url', '')),
        api_key_env=str(provider_cfg.get('api_key_env', 'DASHSCOPE_API_KEY')),
        completion_window=str(batch_cfg.get('completion_window', '24h')),
        poll_interval=int(batch_cfg.get('poll_interval', 60)),
        max_requests_per_batch=int(batch_cfg.get('max_requests_per_batch', 50000)),
        max_line_size_mb=float(batch_cfg.get('max_line_size_mb', 6.0)),
    )


def _evaluate_image(image_path: str, client: Optional[VisionClient]) -> Tuple[bool, Dict[str, Any]]:
    evaluation = None
    keep = True
    if client is not None:
        reply = client.analyse_character(image_path)
        evaluation = _parse_ark_response(reply.get('raw', ''))
        evaluation['response_time'] = reply.get('response_time')
        character = evaluation.get('character')
        keep = (
            bool(evaluation.get('single_character')) and
            bool(evaluation.get('recognizable')) and
            not bool(evaluation.get('has_noise')) and
            isinstance(character, str) and len(character) == 1 and '\u4e00' <= character <= '\u9fff'
        )
    return keep, evaluation or {}


def filter_character_dir(input_dir: str, output_dir: str,
                         vision_cfg: Optional[VisionConfig],
                         force: bool = False) -> Dict[str, Any]:
    ensure_dir(output_dir)
    char_files = [f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('char_')]
    if not char_files:
        return {'total_files': 0}
    char_files.sort()

    result_path = os.path.join(output_dir, 'postocr_results.json')
    if not force and os.path.isfile(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as handle:
                cached = json.load(handle)
            stats = cached.get('stats', {})
            stats['cached'] = True
            return stats
        except Exception:
            pass

    client = VisionClient(vision_cfg) if vision_cfg is not None else None
    stats = {
        'total_files': len(char_files),
        'accepted': 0,
        'filtered_out': 0,
        'error': 0,
    }
    entries: List[Dict[str, Any]] = []

    for idx, filename in enumerate(char_files, 1):
        src_path = os.path.join(input_dir, filename)
        record: Dict[str, Any] = {'original_file': filename}
        try:
            keep, evaluation = _evaluate_image(src_path, client)
            character = evaluation.get('character') if evaluation else None
            if keep and character:
                safe_char = _safe_filename(character)
                dst_name = f"{idx:03d}_{safe_char}.png"
                shutil.copy2(src_path, os.path.join(output_dir, dst_name))
                stats['accepted'] += 1
                record.update({'status': 'accepted', 'output_file': dst_name})
            else:
                dst_name = f"reject_{idx:03d}.png"
                shutil.copy2(src_path, os.path.join(output_dir, dst_name))
                stats['filtered_out'] += 1
                record.update({'status': 'filtered_out', 'output_file': dst_name})
            record['ark_evaluation'] = evaluation
        except Exception as exc:
            stats['error'] += 1
            record.update({'status': 'error', 'error': str(exc)})
        entries.append(record)

    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump({'stats': stats, 'results': entries}, handle, ensure_ascii=False, indent=2)
    return stats


def filter_character_dir_batch(input_dir: str, output_dir: str,
                                batch_client: BatchClient,
                                provider_cfg: Dict[str, Any],
                                system_prompt: str,
                                force: bool = False) -> Dict[str, Any]:
    """批量推理模式处理单个字符目录"""
    ensure_dir(output_dir)
    char_files = [f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('char_')]
    if not char_files:
        return {'total_files': 0}
    char_files.sort()

    result_path = os.path.join(output_dir, 'postocr_results.json')
    if not force and os.path.isfile(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as handle:
                cached = json.load(handle)
            stats = cached.get('stats', {})
            stats['cached'] = True
            return stats
        except Exception:
            pass

    # 收集所有图片路径
    image_paths = [os.path.join(input_dir, f) for f in char_files]
    logger.info(f"批量推理: {len(image_paths)} 张图片")

    # 提取模型配置
    model = str(provider_cfg.get('model', ''))
    enable_thinking_cfg = provider_cfg.get('enable_thinking', 'auto')

    # 处理 enable_thinking 配置
    if enable_thinking_cfg == 'auto':
        # auto 模式：qwen3-vl-plus 自动启用 thinking
        enable_thinking = 'qwen3-vl-plus' in model.lower()
    else:
        # 强制模式：true/false
        enable_thinking = bool(enable_thinking_cfg)

    thinking_budget = int(provider_cfg.get('thinking_budget', 8192))
    max_tokens = int(provider_cfg.get('max_tokens', 512))
    temperature = float(provider_cfg.get('temperature', 0.0))

    # 执行批量推理
    work_dir = os.path.join(output_dir, 'batch_work')
    try:
        batch_result = batch_client.run_batch_inference(
            image_paths=image_paths,
            system_prompt=system_prompt,
            model=model,
            work_dir=work_dir,
            task_name=os.path.basename(input_dir),
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        return {'total_files': len(char_files), 'error': str(e)}

    # 处理结果
    stats = {
        'total_files': len(char_files),
        'accepted': 0,
        'filtered_out': 0,
        'error': 0,
    }
    entries: List[Dict[str, Any]] = []

    # 建立 custom_id -> 原始文件名的映射
    filename_map = {os.path.basename(p): p for p in image_paths}

    for item in batch_result['results']:
        custom_id = item['custom_id']
        original_file = os.path.basename(filename_map.get(custom_id, custom_id))
        src_path = filename_map.get(custom_id)

        record: Dict[str, Any] = {'original_file': original_file}

        if item['error']:
            stats['error'] += 1
            record.update({'status': 'error', 'error': item['error']})
            entries.append(record)
            continue

        # 解析 LLM 响应
        content = item.get('content', '')
        evaluation = _parse_ark_response(content)
        evaluation['response_time'] = 0  # 批量模式无单个响应时间

        character = evaluation.get('character')
        keep = (
            bool(evaluation.get('single_character')) and
            bool(evaluation.get('recognizable')) and
            not bool(evaluation.get('has_noise')) and
            isinstance(character, str) and len(character) == 1 and '\u4e00' <= character <= '\u9fff'
        )

        idx = char_files.index(original_file) + 1

        if keep and character:
            safe_char = _safe_filename(character)
            dst_name = f"{idx:03d}_{safe_char}.png"
            shutil.copy2(src_path, os.path.join(output_dir, dst_name))
            stats['accepted'] += 1
            record.update({'status': 'accepted', 'output_file': dst_name})
        else:
            dst_name = f"reject_{idx:03d}.png"
            shutil.copy2(src_path, os.path.join(output_dir, dst_name))
            stats['filtered_out'] += 1
            record.update({'status': 'filtered_out', 'output_file': dst_name})

        record['ark_evaluation'] = evaluation
        entries.append(record)

    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump({'stats': stats, 'results': entries, 'batch_id': batch_result.get('batch_id')},
                  handle, ensure_ascii=False, indent=2)
    return stats


def process_all_segment_results(input_base_dir: str | None = None,
                                output_base_dir: str | None = None,
                                config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    input_base_dir = input_base_dir or SEGMENTS_DIR
    output_base_dir = output_base_dir or POSTOCR_DIR
    if not os.path.exists(input_base_dir):
        print(f"输入目录不存在: {input_base_dir}")
        return {'error': 'input_dir_not_found'}

    candidate_dirs: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(input_base_dir):
        # 过滤掉不需要处理的目录
        dirs[:] = [d for d in dirs if d not in ('debug', '_overlays')]
        if any(f.startswith('char_') and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            rel_path = os.path.relpath(root, input_base_dir)
            # 双重检查：确保路径中不包含这些目录
            skip_dirs = {'debug', '_overlays'}
            if any(part in skip_dirs for part in rel_path.split(os.sep)):
                continue
            candidate_dirs.append((root, rel_path))

    if not candidate_dirs:
        print(f"在 {input_base_dir} 中未找到包含字符图片的文件夹")
        return {'total_dirs': 0}

    force = bool(config and config.get('force'))
    mode = str(POSTOCR_CONFIG.get('mode', 'realtime')).lower()
    provider = str(POSTOCR_CONFIG.get('provider', 'qwen')).lower()
    provider_cfg = POSTOCR_CONFIG.get(provider, {})

    batch = {
        'total_dirs': len(candidate_dirs),
        'processed_dirs': 0,
        'total_files': 0,
        'total_accepted': 0,
        'total_filtered': 0,
        'total_errors': 0,
    }

    def _print_progress():
        msg = (f"进度: {batch['processed_dirs']}/{batch['total_dirs']} | "
               f"files={batch['total_files']} √={batch['total_accepted']} ✗={batch['total_filtered']} err={batch['total_errors']}")
        print(f"\r{msg}", end='', flush=True)

    pending: List[Tuple[str, str]] = []
    for abs_dir, rel_dir in candidate_dirs:
        out_dir = os.path.join(output_base_dir, rel_dir)
        result_path = os.path.join(out_dir, 'postocr_results.json')
        if not force and os.path.isfile(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as handle:
                    cached = json.load(handle)
                stats = cached.get('stats', {})
                batch['processed_dirs'] += 1
                batch['total_files'] += stats.get('total_files', 0)
                batch['total_accepted'] += stats.get('accepted', 0)
                batch['total_filtered'] += stats.get('filtered_out', 0)
                batch['total_errors'] += stats.get('error', 0)
                print(f"[skip] {rel_dir} 已存在 postocr_results.json")
                _print_progress()
                continue
            except Exception:
                pass
        pending.append((abs_dir, rel_dir))

    print("\n=== POSTOCR 过滤开始 ===")
    print(f"输入目录: {input_base_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"模式: {mode.upper()}")
    print(f"提供商: {provider.upper()}")

    # 批量模式
    if mode == 'batch':
        # 按"图"分组（例如 01a_preprocessed, 01b_preprocessed）
        from collections import defaultdict
        image_groups = defaultdict(list)  # key: 图名称, value: [(abs_dir, rel_dir), ...]

        for abs_dir, rel_dir in pending:
            # 提取图名称（第一级目录）
            parts = rel_dir.split(os.sep)
            image_name = parts[0] if parts else 'misc'
            image_groups[image_name].append((abs_dir, rel_dir))

        print(f"找到 {len(candidate_dirs)} 个字符目录，分为 {len(image_groups)} 个图，其中 {len(pending)} 个待处理")
        print(f"批量模式：一次性提交所有图的任务\n")

        batch_cfg = _build_batch_config()
        if not batch_cfg:
            print("批量推理配置构建失败")
            return batch

        batch_client = BatchClient(batch_cfg)
        system_prompt = str(provider_cfg.get('system_prompt', VisionConfig.system_prompt))

        # 提取模型配置
        model = str(provider_cfg.get('model', ''))
        enable_thinking_cfg = provider_cfg.get('enable_thinking', 'auto')
        if enable_thinking_cfg == 'auto':
            enable_thinking = 'qwen3-vl-plus' in model.lower()
        else:
            enable_thinking = bool(enable_thinking_cfg)
        thinking_budget = int(provider_cfg.get('thinking_budget', 8192))
        max_tokens = int(provider_cfg.get('max_tokens', 512))
        temperature = float(provider_cfg.get('temperature', 0.0))

        # 为每个图创建批量任务（不等待）
        batch_tasks = []  # [(image_name, batch_id, work_dir, dirs), ...]

        for image_name, dirs in sorted(image_groups.items()):
            # 创建该图的工作目录
            image_work_dir = os.path.join(output_base_dir, image_name, '_batch_work')
            os.makedirs(image_work_dir, exist_ok=True)

            # 检查是否已有批量任务（断点恢复）
            existing_state = batch_client.load_batch_state(image_work_dir)
            if existing_state and not force:
                batch_id = existing_state['batch_id']
                status = existing_state.get('status', 'unknown')

                # 如果任务还在处理中，跳过
                if status in ('submitted', 'validating', 'in_progress', 'finalizing'):
                    print(f"[跳过] {image_name}: 批量任务 {batch_id} 已存在（状态: {status}）")
                    batch_tasks.append({
                        'image_name': image_name,
                        'batch_id': batch_id,
                        'work_dir': image_work_dir,
                        'total_files': existing_state.get('total_files', 0),
                        'dirs': dirs,
                        'skipped': True
                    })
                    continue

                # 如果已完成，检查是否需要重新处理
                elif status == 'completed':
                    print(f"[跳过] {image_name}: 批量任务 {batch_id} 已完成")
                    batch_tasks.append({
                        'image_name': image_name,
                        'batch_id': batch_id,
                        'work_dir': image_work_dir,
                        'total_files': existing_state.get('total_files', 0),
                        'dirs': dirs,
                        'skipped': True
                    })
                    continue

            # 收集该图的所有字符文件
            all_image_paths = []
            all_custom_ids = []  # 唯一标识列表
            char_file_mapping = []  # [(image_path, abs_dir, rel_dir, custom_id), ...]

            for abs_dir, rel_dir in dirs:
                char_files = [f for f in os.listdir(abs_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('char_')]
                for cf in char_files:
                    img_path = os.path.join(abs_dir, cf)
                    # 使用相对路径作为唯一 custom_id（例如：region_001/char_0001.png）
                    custom_id = os.path.join(rel_dir.replace(image_name + os.sep, ''), cf)
                    all_image_paths.append(img_path)
                    all_custom_ids.append(custom_id)
                    char_file_mapping.append((img_path, abs_dir, rel_dir, custom_id))

            if not all_image_paths:
                continue

            print(f"[提交] {image_name}: {len(all_image_paths)} 个字符图片...")

            try:
                # 生成 JSONL（使用唯一的 custom_ids）
                jsonl_path = os.path.join(image_work_dir, 'batch_input.jsonl')
                success_count, skipped_count = batch_client.generate_jsonl(
                    image_paths=all_image_paths,
                    system_prompt=system_prompt,
                    model=model,
                    output_path=jsonl_path,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    custom_ids=all_custom_ids  # 传入唯一标识
                )

                if success_count == 0:
                    logger.error(f"{image_name}: 没有成功生成任何请求")
                    continue

                # 上传文件
                file_id = batch_client.upload_file(jsonl_path)

                # 创建批量任务
                batch_id = batch_client.create_batch(file_id, task_name=image_name)

                # 保存映射信息（包含状态追踪）
                mapping_file = os.path.join(image_work_dir, 'file_mapping.json')
                current_time = time.time()
                with open(mapping_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'image_name': image_name,
                        'total_files': len(all_image_paths),
                        'first_submitted': current_time,
                        'last_updated': current_time,
                        'batch_history': [
                            {'batch_id': batch_id, 'type': 'initial', 'timestamp': current_time}
                        ],
                        'dirs': dirs,
                        'mapping': [
                            {
                                'image_path': p,
                                'abs_dir': a,
                                'rel_dir': r,
                                'custom_id': c,
                                'status': 'submitted',
                                'first_batch_id': batch_id,
                                'last_batch_id': batch_id,
                            }
                            for p, a, r, c in char_file_mapping
                        ]
                    }, f, ensure_ascii=False, indent=2)

                # 保存状态（包含文件数量信息）
                state_file = os.path.join(image_work_dir, 'batch_state.json')
                state = {
                    'batch_id': batch_id,
                    'status': 'submitted',
                    'timestamp': time.time(),
                    'total_files': len(all_image_paths),
                    'image_name': image_name,
                }
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)

                batch_tasks.append({
                    'image_name': image_name,
                    'batch_id': batch_id,
                    'work_dir': image_work_dir,
                    'total_files': len(all_image_paths),
                    'dirs': dirs
                })

                print(f"  ✓ 批量任务已创建: {batch_id}")

            except Exception as e:
                logger.error(f"{image_name}: 创建批量任务失败: {e}")
                continue

        new_tasks = [t for t in batch_tasks if not t.get('skipped')]
        skipped_tasks = [t for t in batch_tasks if t.get('skipped')]

        print(f"\n=== 批量任务提交汇总 ===")
        print(f"新提交: {len(new_tasks)} 个任务")
        print(f"已存在: {len(skipped_tasks)} 个任务")
        print(f"总计: {len(batch_tasks)} 个任务\n")

        for task in new_tasks:
            print(f"  ✓ [新] {task['image_name']}: {task['batch_id']} ({task['total_files']} 个文件)")
        for task in skipped_tasks:
            print(f"  - [跳过] {task['image_name']}: {task['batch_id']}")

        # 保存/更新任务信息（合并已有任务）
        tasks_info_file = os.path.join(output_base_dir, '_batch_tasks.json')

        # 读取现有任务
        existing_tasks = {}
        if os.path.exists(tasks_info_file):
            try:
                with open(tasks_info_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for task in existing_data.get('tasks', []):
                        existing_tasks[task['image_name']] = task
            except Exception as e:
                logger.warning(f"读取已有任务信息失败: {e}")

        # 合并任务（新任务覆盖旧任务）
        for task in batch_tasks:
            existing_tasks[task['image_name']] = task

        # 保存合并后的任务列表
        with open(tasks_info_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'input_dir': input_base_dir,
                'total_tasks': len(existing_tasks),
                'tasks': list(existing_tasks.values())
            }, f, ensure_ascii=False, indent=2)

        print(f"\n任务信息已保存: {tasks_info_file}")
        print(f"\n提示：")
        print(f"  - 查询任务状态: ./pipeline postocr status <batch_id>")
        print(f"  - 等待并下载结果: ./pipeline postocr wait {output_base_dir}")

        return batch

    # 实时模式
    else:
        workers = max(1, int(POSTOCR_CONFIG.get('workers', 4)))
        print(f"找到 {len(candidate_dirs)} 个字符目录，其中 {len(pending)} 个待处理，使用 {workers} 个 worker")

        vision_cfg = _build_vision_config()

        def _task(pair: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
            abs_dir, rel_dir = pair
            out_dir = os.path.join(output_base_dir, rel_dir)
            stats = filter_character_dir(abs_dir, out_dir, vision_cfg, force=force)
            return rel_dir, stats

        if pending:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_dir = {executor.submit(_task, pair): pair[1] for pair in pending}
                for future in as_completed(future_to_dir):
                    rel_dir = future_to_dir[future]
                    try:
                        _, stats = future.result()
                        batch['processed_dirs'] += 1
                        batch['total_files'] += stats.get('total_files', 0)
                        batch['total_accepted'] += stats.get('accepted', 0)
                        batch['total_filtered'] += stats.get('filtered_out', 0)
                        batch['total_errors'] += stats.get('error', 0)
                        print(f"[done] {rel_dir}: {stats.get('accepted', 0)}/{stats.get('total_files', 0)}")
                        _print_progress()
                    except Exception as exc:
                        logger.error("目录 %s 处理失败: %s", rel_dir, exc)
                        batch['total_errors'] += 1
                        _print_progress()
    _print_progress()
    print()
    print("=== POSTOCR 过滤完成 ===")
    print(f"处理目录数: {batch['processed_dirs']}/{batch['total_dirs']}")
    print(f"总文件数: {batch['total_files']}")
    print(f"通过过滤: {batch['total_accepted']}")
    print(f"过滤淘汰: {batch['total_filtered']}")
    print(f"总错误: {batch['total_errors']}")

    return batch


def wait_and_process_batch_tasks(output_base_dir: str) -> Dict[str, Any]:
    """等待所有批量任务完成并处理结果"""
    tasks_info_file = os.path.join(output_base_dir, '_batch_tasks.json')
    if not os.path.exists(tasks_info_file):
        raise FileNotFoundError(f"未找到批量任务信息文件: {tasks_info_file}")

    with open(tasks_info_file, 'r', encoding='utf-8') as f:
        tasks_info = json.load(f)

    batch_tasks = tasks_info['tasks']
    print(f"\n=== 等待 {len(batch_tasks)} 个批量任务完成 ===\n")

    # 构建批量客户端
    batch_cfg = _build_batch_config()
    if not batch_cfg:
        raise RuntimeError("批量推理配置构建失败")

    batch_client = BatchClient(batch_cfg)

    # 等待所有任务完成
    completed_tasks = []
    for idx, task in enumerate(batch_tasks, 1):
        image_name = task['image_name']
        batch_id = task['batch_id']
        work_dir = task['work_dir']

        print(f"[{idx}/{len(batch_tasks)}] {image_name} ({batch_id})")

        try:
            # 等待完成
            status_info = batch_client.wait_for_completion(batch_id)

            # 下载结果
            result_jsonl = os.path.join(work_dir, 'batch_results.jsonl')
            if status_info['output_file_id']:
                batch_client.download_results(status_info['output_file_id'], result_jsonl)

            # 下载错误文件
            if status_info['error_file_id']:
                error_jsonl = os.path.join(work_dir, 'batch_errors.jsonl')
                batch_client.download_results(status_info['error_file_id'], error_jsonl)

            # 更新状态
            batch_client.save_batch_state(work_dir, batch_id, status=status_info['status'])

            completed_tasks.append({
                **task,
                'status': status_info['status'],
                'result_file': result_jsonl if status_info['output_file_id'] else None,
                'request_counts': status_info['request_counts']
            })

            print(f"  ✓ 完成: {status_info['request_counts']['completed']}/{status_info['request_counts']['total']}")

        except Exception as e:
            logger.error(f"{image_name}: 处理失败: {e}")
            completed_tasks.append({
                **task,
                'status': 'error',
                'error': str(e)
            })

    # 处理所有结果
    print(f"\n=== 处理下载的结果 ===\n")

    total_stats = {
        'total_tasks': len(completed_tasks),
        'success_count': 0,
        'total_files': 0,
        'total_accepted': 0,
        'total_filtered': 0,
        'total_errors': 0
    }

    for task in completed_tasks:
        if task.get('status') != 'completed' or not task.get('result_file'):
            continue

        image_name = task['image_name']
        work_dir = task['work_dir']
        result_file = task['result_file']

        print(f"[处理] {image_name}")

        try:
            # 读取映射信息
            mapping_file = os.path.join(work_dir, 'file_mapping.json')
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_info = json.load(f)

            # 解析结果
            results = batch_client.parse_results(result_file)

            # 建立 custom_id -> 映射的字典
            id_to_mapping = {
                m.get('custom_id', os.path.basename(m['image_path'])): m
                for m in mapping_info['mapping']
            }

            # 按目录分组处理结果
            from collections import defaultdict
            results_by_dir = defaultdict(list)

            for item in results:
                custom_id = item['custom_id']
                mapping = id_to_mapping.get(custom_id)
                if not mapping:
                    logger.warning(f"未找到映射: {custom_id}")
                    continue

                rel_dir = mapping['rel_dir']
                abs_dir = mapping['abs_dir']
                results_by_dir[rel_dir].append((item, abs_dir, mapping['image_path']))

            # 为每个目录处理结果
            for rel_dir, dir_results in results_by_dir.items():
                out_dir = os.path.join(output_base_dir, rel_dir)
                ensure_dir(out_dir)

                stats = {
                    'total_files': len(dir_results),
                    'accepted': 0,
                    'filtered_out': 0,
                    'error': 0
                }
                entries = []

                for item, abs_dir, image_path in dir_results:
                    original_file = os.path.basename(image_path)
                    src_path = image_path
                    record = {'original_file': original_file}

                    # 从原始文件名提取编号（例如 char_001.png → 001）
                    try:
                        file_num = ''.join(filter(str.isdigit, original_file.split('.')[0]))
                        file_idx = int(file_num) if file_num else 0
                    except:
                        file_idx = 0

                    if item.get('error'):
                        stats['error'] += 1
                        record.update({'status': 'error', 'error': item['error']})
                        entries.append(record)
                        continue

                    # 解析 LLM 响应
                    content = item.get('content', '')
                    evaluation = _parse_ark_response(content)
                    evaluation['response_time'] = 0

                    character = evaluation.get('character')
                    keep = (
                        bool(evaluation.get('single_character')) and
                        bool(evaluation.get('recognizable')) and
                        not bool(evaluation.get('has_noise')) and
                        isinstance(character, str) and len(character) == 1 and '\u4e00' <= character <= '\u9fff'
                    )

                    # 使用原文件编号，保持增量更新时编号稳定
                    if keep and character:
                        safe_char = _safe_filename(character)
                        dst_name = f"{file_idx:03d}_{safe_char}.png" if file_idx > 0 else f"{safe_char}.png"
                        dst_path = os.path.join(out_dir, dst_name)
                        # 只有文件不存在时才复制（避免重复）
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
                        stats['accepted'] += 1
                        record.update({'status': 'accepted', 'output_file': dst_name})
                    else:
                        dst_name = f"reject_{file_idx:03d}.png" if file_idx > 0 else "reject.png"
                        dst_path = os.path.join(out_dir, dst_name)
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
                        stats['filtered_out'] += 1
                        record.update({'status': 'filtered_out', 'output_file': dst_name})

                    record['ark_evaluation'] = evaluation
                    entries.append(record)

                # 增量保存结果（字符级粒度，支持重试）
                result_path = os.path.join(out_dir, 'postocr_results.json')
                merged_stats = _save_incremental_results(result_path, entries, task['batch_id'])

                # 使用合并后的统计（包含已有结果）
                total_stats['total_files'] += merged_stats['total_files']
                total_stats['total_accepted'] += merged_stats['accepted']
                total_stats['total_filtered'] += merged_stats['filtered_out']
                total_stats['total_errors'] += merged_stats['error']

            # 更新 file_mapping.json 中的状态
            # 1. 读取错误文件（如果存在）
            error_file = os.path.join(work_dir, 'batch_errors.jsonl')
            error_map = {}  # custom_id -> error_info
            if os.path.exists(error_file):
                try:
                    with open(error_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            error_item = json.loads(line)
                            custom_id = error_item.get('custom_id')
                            error_info = error_item.get('error', {})
                            error_map[custom_id] = {
                                'code': error_info.get('code', 'unknown'),
                                'message': error_info.get('message', ''),
                                'timestamp': time.time()
                            }
                except Exception as e:
                    logger.warning(f"读取错误文件失败: {e}")

            # 2. 建立成功结果的 custom_id 集合
            success_ids = {item['custom_id'] for item in results}

            # 3. 更新映射信息中的状态
            for mapping_entry in mapping_info['mapping']:
                custom_id = mapping_entry['custom_id']

                if custom_id in success_ids:
                    # 成功处理
                    mapping_entry['status'] = 'success'
                    mapping_entry['last_batch_id'] = task['batch_id']
                elif custom_id in error_map:
                    # 处理失败
                    mapping_entry['status'] = 'failed'
                    mapping_entry['last_batch_id'] = task['batch_id']

                    # 添加到错误历史
                    if 'error_history' not in mapping_entry:
                        mapping_entry['error_history'] = []

                    error_info = error_map[custom_id]
                    mapping_entry['error_history'].append({
                        'batch_id': task['batch_id'],
                        'error_code': error_info['code'],
                        'error_msg': error_info['message'],
                        'timestamp': error_info['timestamp']
                    })

            # 4. 更新 file_mapping.json
            mapping_info['last_updated'] = time.time()
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_info, f, ensure_ascii=False, indent=2)

            total_stats['success_count'] += 1
            print(f"  ✓ 已处理 {len(results)} 个结果，更新了 file_mapping.json")

        except Exception as e:
            logger.error(f"{image_name}: 结果处理失败: {e}")
            import traceback
            traceback.print_exc()

    return total_stats


def retry_failed_requests(output_base_dir: str, mode: str = 'batch') -> Dict[str, Any]:
    """收集并重试失败的批量请求

    Args:
        output_base_dir: PostOCR输出目录
        mode: 'batch' 或 'realtime'

    Returns:
        统计信息
    """
    from PIL import Image

    print(f"\n=== 收集失败的请求 ===")

    # 收集所有 _batch_work 目录
    batch_work_dirs = []
    for root, dirs, files in os.walk(output_base_dir):
        if os.path.basename(root) == '_batch_work':
            batch_work_dirs.append(root)

    if not batch_work_dirs:
        print("未找到任何批量任务目录")
        return {'total_failed': 0}

    print(f"找到 {len(batch_work_dirs)} 个批量任务目录\n")

    # 收集所有失败的请求
    failed_requests = []  # [(custom_id, error_code, error_msg, image_path, abs_dir, rel_dir), ...]

    for work_dir in batch_work_dirs:
        error_file = os.path.join(work_dir, 'batch_errors.jsonl')
        if not os.path.exists(error_file):
            continue

        # 读取映射信息
        mapping_file = os.path.join(work_dir, 'file_mapping.json')
        if not os.path.exists(mapping_file):
            logger.warning(f"未找到映射文件: {mapping_file}")
            continue

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_info = json.load(f)

        # 建立 custom_id -> 映射的字典
        id_to_mapping = {
            m.get('custom_id', os.path.basename(m['image_path'])): m
            for m in mapping_info['mapping']
        }

        # 读取错误文件
        with open(error_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    error_item = json.loads(line)
                    custom_id = error_item.get('custom_id')
                    error_info = error_item.get('error', {})
                    error_code = error_info.get('code', 'unknown')
                    error_msg = error_info.get('message', '')

                    # 获取映射信息
                    mapping = id_to_mapping.get(custom_id)
                    if not mapping:
                        logger.warning(f"未找到映射: {custom_id}")
                        continue

                    failed_requests.append({
                        'custom_id': custom_id,
                        'error_code': error_code,
                        'error_msg': error_msg,
                        'image_path': mapping['image_path'],
                        'abs_dir': mapping['abs_dir'],
                        'rel_dir': mapping['rel_dir']
                    })
                except Exception as e:
                    logger.warning(f"解析错误行失败: {e}")
                    continue

    print(f"收集到 {len(failed_requests)} 个失败请求")

    if not failed_requests:
        return {'total_failed': 0, 'filtered_small': 0, 'retry_submitted': 0}

    # 统计错误类型
    error_stats = {}
    for req in failed_requests:
        code = req['error_code']
        error_stats[code] = error_stats.get(code, 0) + 1

    print("\n错误分类:")
    for code, count in sorted(error_stats.items(), key=lambda x: -x[1]):
        print(f"  {code}: {count} 个")

    # 过滤掉小图片（InvalidParameter 错误）
    print("\n=== 过滤小图片 ===")
    retry_list = []
    filtered_small = 0

    for req in failed_requests:
        # 检查图片尺寸
        try:
            img = Image.open(req['image_path'])
            w, h = img.size

            if w < 10 or h < 10:
                filtered_small += 1
                print(f"  [过滤] {os.path.basename(req['image_path'])}: {w}x{h}px (太小)")
                continue
        except Exception as e:
            logger.warning(f"无法读取图片 {req['image_path']}: {e}")
            continue

        retry_list.append(req)

    print(f"\n过滤掉 {filtered_small} 个小图片")
    print(f"待验证: {len(retry_list)} 个请求")

    # 验证实际状态（过滤已被实时重试成功的）
    print("\n=== 验证实际状态（支持混合使用批量+实时）===")
    actual_retry_list = []
    already_success = 0

    for req in retry_list:
        rel_dir = req['rel_dir']
        original_file = os.path.basename(req['image_path'])

        # 检查 postocr_results.json 的实际状态
        result_path = os.path.join(output_base_dir, rel_dir, 'postocr_results.json')
        existing = _load_existing_results(result_path)
        entry = existing['results_map'].get(original_file)

        if entry:
            status = entry.get('status')
            # 只有真正失败的才需要重试（postocr_results.json 只使用 'error'）
            if status == 'error':
                actual_retry_list.append(req)
            elif status in ('accepted', 'filtered_out'):
                # 已成功处理（批量或实时），跳过
                already_success += 1
                print(f"  [跳过] {original_file}: 已成功处理（状态: {status}）")
            else:
                # 未知状态，保守重试
                actual_retry_list.append(req)
        else:
            # postocr_results.json 中不存在（理论上不应该），重试
            actual_retry_list.append(req)

    print(f"\n状态验证完成:")
    print(f"  批量失败: {len(failed_requests)} 个")
    print(f"  过滤小图: {filtered_small} 个")
    print(f"  已被实时重试成功: {already_success} 个")
    print(f"  实际需要重试: {len(actual_retry_list)} 个\n")

    if not actual_retry_list:
        return {
            'total_failed': len(failed_requests),
            'filtered_small': filtered_small,
            'already_success': already_success,
            'retry_submitted': 0
        }

    # 根据模式处理重试
    if mode == 'batch':
        return _retry_with_batch(output_base_dir, actual_retry_list, filtered_small, len(failed_requests))
    else:
        return _retry_with_realtime(output_base_dir, actual_retry_list, filtered_small, len(failed_requests))


def _retry_with_batch(output_base_dir: str, retry_list: List[Dict[str, Any]],
                     filtered_small: int, total_failed: int) -> Dict[str, Any]:
    """使用批量模式重试"""
    print("=== 批量重试模式 ===")

    # 按图分组
    from collections import defaultdict
    image_groups = defaultdict(list)

    for req in retry_list:
        # 提取图名称（第一级目录）
        parts = req['rel_dir'].split(os.sep)
        image_name = parts[0] if parts else 'misc'
        image_groups[image_name].append(req)

    print(f"需要重试 {len(retry_list)} 个请求，分为 {len(image_groups)} 个图\n")

    # 构建批量客户端
    batch_cfg = _build_batch_config()
    if not batch_cfg:
        raise RuntimeError("批量推理配置构建失败")

    batch_client = BatchClient(batch_cfg)

    # 获取provider配置
    provider = str(POSTOCR_CONFIG.get('provider', 'qwen')).lower()
    provider_cfg = POSTOCR_CONFIG.get(provider, {})
    system_prompt = str(provider_cfg.get('system_prompt', VisionConfig.system_prompt))

    # 提取模型配置
    model = str(provider_cfg.get('model', ''))
    enable_thinking_cfg = provider_cfg.get('enable_thinking', 'auto')
    if enable_thinking_cfg == 'auto':
        enable_thinking = 'qwen3-vl-plus' in model.lower()
    else:
        enable_thinking = bool(enable_thinking_cfg)
    thinking_budget = int(provider_cfg.get('thinking_budget', 8192))
    max_tokens = int(provider_cfg.get('max_tokens', 512))
    temperature = float(provider_cfg.get('temperature', 0.0))

    # 为每个图创建重试任务
    retry_tasks = []

    for image_name, requests in sorted(image_groups.items()):
        # 创建重试工作目录
        retry_work_dir = os.path.join(output_base_dir, image_name, '_batch_work')
        os.makedirs(retry_work_dir, exist_ok=True)

        # 收集图片路径和 custom_ids
        image_paths = [req['image_path'] for req in requests]
        custom_ids = [req['custom_id'] for req in requests]

        print(f"[提交] {image_name}: 重试 {len(requests)} 个失败请求...")

        try:
            # 读取现有的 file_mapping.json（保留历史）
            mapping_file = os.path.join(retry_work_dir, 'file_mapping.json')
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_info = json.load(f)
            else:
                # 如果不存在（理论上不应该发生），创建新的
                mapping_info = {
                    'image_name': image_name,
                    'total_files': 0,
                    'first_submitted': time.time(),
                    'batch_history': [],
                    'mapping': []
                }

            # 建立 custom_id -> mapping 的字典（用于快速查找）
            mapping_dict = {m['custom_id']: m for m in mapping_info['mapping']}

            # 更新失败文件的状态为 "retrying"
            retry_custom_ids = {req['custom_id'] for req in requests}
            for custom_id in retry_custom_ids:
                if custom_id in mapping_dict:
                    mapping_dict[custom_id]['status'] = 'retrying'
            # 生成 JSONL（使用原来的 custom_ids）
            jsonl_path = os.path.join(retry_work_dir, 'retry_batch_input.jsonl')
            success_count, skipped_count = batch_client.generate_jsonl(
                image_paths=image_paths,
                system_prompt=system_prompt,
                model=model,
                output_path=jsonl_path,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                temperature=temperature,
                custom_ids=custom_ids
            )

            if success_count == 0:
                logger.error(f"{image_name}: 没有成功生成任何请求")
                continue

            # 上传文件
            file_id = batch_client.upload_file(jsonl_path)

            # 创建批量任务
            batch_id = batch_client.create_batch(file_id, task_name=f"{image_name}_retry")

            # 保存/更新状态（追加到现有状态）
            state_file = os.path.join(retry_work_dir, 'batch_state.json')

            # 读取现有状态
            existing_state = {}
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        existing_state = json.load(f)
                except:
                    pass

            # 创建新状态（保留旧的 batch_id）
            retry_state = {
                'batch_id': batch_id,
                'status': 'submitted',
                'timestamp': time.time(),
                'total_files': len(requests),
                'image_name': image_name,
                'is_retry': True,
                'previous_batch_id': existing_state.get('batch_id')
            }

            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(retry_state, f, ensure_ascii=False, indent=2)

            # 更新 file_mapping.json（添加 batch_history 并保存）
            current_time = time.time()

            # 添加新的 batch 到历史记录
            if 'batch_history' not in mapping_info:
                mapping_info['batch_history'] = []
            mapping_info['batch_history'].append({
                'batch_id': batch_id,
                'type': 'retry',
                'timestamp': current_time
            })

            # 更新时间戳
            mapping_info['last_updated'] = current_time

            # 更新重试文件的 last_batch_id
            for custom_id in retry_custom_ids:
                if custom_id in mapping_dict:
                    mapping_dict[custom_id]['last_batch_id'] = batch_id

            # 重新构建完整的 mapping 列表（保留所有文件，包括成功的）
            # 但只有重试的文件会在本次 wait 中被更新
            all_mapping = list(mapping_dict.values())

            # 保存更新后的 file_mapping.json（保留所有历史）
            # total_files 保持为所有文件的数量（不只是重试的）
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_name': mapping_info['image_name'],
                    'total_files': len(all_mapping),  # 所有文件（包括成功的）
                    'retry_files': len(retry_custom_ids),  # 本次重试的文件数
                    'first_submitted': mapping_info.get('first_submitted', current_time),
                    'last_updated': current_time,
                    'batch_history': mapping_info['batch_history'],
                    'mapping': all_mapping
                }, f, ensure_ascii=False, indent=2)

            retry_tasks.append({
                'image_name': image_name,
                'batch_id': batch_id,
                'work_dir': retry_work_dir,
                'total_files': len(requests)
            })

            print(f"  ✓ 重试批量任务已创建: {batch_id}")

        except Exception as e:
            logger.error(f"{image_name}: 创建重试批量任务失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 更新任务信息
    tasks_info_file = os.path.join(output_base_dir, '_batch_tasks.json')

    # 读取现有任务
    existing_tasks = {}
    if os.path.exists(tasks_info_file):
        try:
            with open(tasks_info_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for task in existing_data.get('tasks', []):
                    existing_tasks[task['image_name']] = task
        except Exception as e:
            logger.warning(f"读取已有任务信息失败: {e}")

    # 更新重试任务（覆盖旧任务）
    for task in retry_tasks:
        existing_tasks[task['image_name']] = task

    # 保存合并后的任务列表
    with open(tasks_info_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.time(),
            'total_tasks': len(existing_tasks),
            'tasks': list(existing_tasks.values())
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== 重试任务提交完成 ===")
    print(f"提交重试任务: {len(retry_tasks)} 个")
    print(f"任务信息已更新: {tasks_info_file}")

    return {
        'total_failed': total_failed,
        'filtered_small': filtered_small,
        'retry_submitted': len(retry_list),
        'retry_tasks': len(retry_tasks)
    }


def _retry_with_realtime(output_base_dir: str, retry_list: List[Dict[str, Any]],
                        filtered_small: int, total_failed: int) -> Dict[str, Any]:
    """使用实时模式重试"""
    print("=== 实时重试模式 ===")

    vision_cfg = _build_vision_config()
    if not vision_cfg:
        raise RuntimeError("Vision 配置构建失败")

    client = VisionClient(vision_cfg)

    success_count = 0
    failed_count = 0
    skipped_count = 0

    # 按目录分组，预加载已有结果（用于断点续传）
    from collections import defaultdict
    requests_by_dir = defaultdict(list)
    for req in retry_list:
        rel_dir = req['rel_dir']
        requests_by_dir[rel_dir].append(req)

    # 预加载所有目录的已有结果
    existing_results = {}  # rel_dir -> {original_file -> status}
    for rel_dir in requests_by_dir.keys():
        result_path = os.path.join(output_base_dir, rel_dir, 'postocr_results.json')
        existing = _load_existing_results(result_path)
        # results_map 格式: {original_file: entry}
        existing_results[rel_dir] = {
            original_file: entry.get('status')
            for original_file, entry in existing['results_map'].items()
        }

    for idx, req in enumerate(retry_list, 1):
        original_file = os.path.basename(req['image_path'])
        rel_dir = req['rel_dir']

        # 检查是否已经成功处理（断点续传）
        existing_status = existing_results.get(rel_dir, {}).get(original_file)
        if existing_status in ('accepted', 'filtered_out'):
            print(f"[{idx}/{len(retry_list)}] {original_file} - 跳过（已处理: {existing_status}）")
            skipped_count += 1
            continue

        print(f"[{idx}/{len(retry_list)}] {original_file}")

        # 找到对应的输出目录和结果文件
        out_dir = os.path.join(output_base_dir, rel_dir)
        result_path = os.path.join(out_dir, 'postocr_results.json')

        # 提取文件编号
        try:
            file_num = ''.join(filter(str.isdigit, original_file.split('.')[0]))
            file_idx = int(file_num) if file_num else 0
        except:
            file_idx = 0

        try:
            keep, evaluation = _evaluate_image(req['image_path'], client)
            character = evaluation.get('character')

            # 创建结果条目
            record = {
                'original_file': original_file,
                'ark_evaluation': evaluation
            }

            if keep and character:
                safe_char = _safe_filename(character)
                dst_name = f"{file_idx:03d}_{safe_char}.png" if file_idx > 0 else f"{safe_char}.png"
                dst_path = os.path.join(out_dir, dst_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(req['image_path'], dst_path)
                record.update({'status': 'accepted', 'output_file': dst_name})
                success_count += 1
                print(f"  ✓ 通过: {character}")
            else:
                dst_name = f"reject_{file_idx:03d}.png" if file_idx > 0 else "reject.png"
                dst_path = os.path.join(out_dir, dst_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(req['image_path'], dst_path)
                record.update({'status': 'filtered_out', 'output_file': dst_name})
                print(f"  ✗ 过滤: {character or '无'}")

            # 增量保存结果
            _save_incremental_results(result_path, [record], batch_id='realtime_retry')

        except Exception as e:
            # 保存错误信息到结果文件
            error_record = {
                'original_file': original_file,
                'status': 'error',
                'error': str(e),
                'retry_failed': True
            }
            _save_incremental_results(result_path, [error_record], batch_id='realtime_retry')

            logger.error(f"处理失败: {e}")
            failed_count += 1
            print(f"  ✗ 错误: {str(e)[:50]}...")

    print(f"\n=== 实时重试完成 ===")
    print(f"跳过（已处理）: {skipped_count}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")

    return {
        'total_failed': total_failed,
        'filtered_small': filtered_small,
        'retry_submitted': len(retry_list),
        'retry_skipped': skipped_count,
        'retry_success': success_count,
        'retry_failed': failed_count
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='基于 Doubao Vision 的字符质量过滤')
    parser.add_argument('--input-dir', help='字符图片目录（包含 char_*.png）')
    parser.add_argument('--output-dir', help='输出目录，默认写到 POSTOCR_DIR 对应位置')
    parser.add_argument('--force', action='store_true', help='忽略已有 postocr_results.json，重新处理')
    args = parser.parse_args()

    if args.input_dir:
        in_dir = args.input_dir
        if not os.path.isdir(in_dir):
            print(f"输入目录不存在: {in_dir}")
            sys.exit(1)
        out_dir = args.output_dir or os.path.join(POSTOCR_DIR, os.path.basename(os.path.normpath(in_dir)))
        stats = filter_character_dir(in_dir, out_dir, _build_vision_config(), force=args.force)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        batch_stats = process_all_segment_results(config={'force': args.force})
        print(json.dumps(batch_stats, ensure_ascii=False, indent=2))
