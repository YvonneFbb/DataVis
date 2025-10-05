"""Post-OCR filtering using Doubao vision model."""
from __future__ import annotations

import json
import os
import re
import logging
import shutil
import sys
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
from src.postocr.ark_client import ArkVisionClient, ArkVisionConfig

logger = logging.getLogger(__name__)


def _safe_filename(text: str, max_length: int = 20) -> str:
    text = (text or '').strip()
    if not text:
        return 'unknown'
    safe = re.sub(r'[<>:"/\\|?*]', '_', text)
    return safe[:max_length]


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


def _build_vision_config() -> Optional[ArkVisionConfig]:
    """根据配置构建视觉模型客户端配置"""
    if not POSTOCR_CONFIG.get('enabled', False):
        return None

    provider = str(POSTOCR_CONFIG.get('provider', 'doubao')).lower()
    provider_cfg = POSTOCR_CONFIG.get(provider, {})

    if not provider_cfg:
        logger.warning(f"未找到 provider '{provider}' 的配置，PostOCR 将被禁用")
        return None

    return ArkVisionConfig(
        base_url=str(provider_cfg.get('base_url', '')),
        model=str(provider_cfg.get('model', '')),
        api_key_env=str(provider_cfg.get('api_key_env', 'ARK_API_KEY')),
        timeout=int(provider_cfg.get('timeout', 60)),
        temperature=float(provider_cfg.get('temperature', 0.0)),
        max_tokens=int(provider_cfg.get('max_tokens', 256)),
        enable_thinking=bool(provider_cfg.get('enable_thinking', False)),
        thinking_budget=int(provider_cfg.get('thinking_budget', 8192)),
    )


def _evaluate_image(image_path: str, client: Optional[ArkVisionClient]) -> Tuple[bool, Dict[str, Any]]:
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
                         ark_cfg: Optional[ArkVisionConfig],
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

    client = ArkVisionClient(ark_cfg) if ark_cfg is not None else None
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
        dirs[:] = [d for d in dirs if d != 'debug']
        if any(f.startswith('char_') and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            rel_path = os.path.relpath(root, input_base_dir)
            if 'debug' in rel_path.split(os.sep):
                continue
            candidate_dirs.append((root, rel_path))

    if not candidate_dirs:
        print(f"在 {input_base_dir} 中未找到包含字符图片的文件夹")
        return {'total_dirs': 0}

    force = bool(config and config.get('force'))
    ark_cfg = _build_vision_config()
    workers = max(1, int(POSTOCR_CONFIG.get('workers', 4)))

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
    print(f"找到 {len(candidate_dirs)} 个字符目录，其中 {len(pending)} 个待处理，使用 {workers} 个 worker")

    def _task(pair: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
        abs_dir, rel_dir = pair
        out_dir = os.path.join(output_base_dir, rel_dir)
        stats = filter_character_dir(abs_dir, out_dir, ark_cfg, force=force)
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
