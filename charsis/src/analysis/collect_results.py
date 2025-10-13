#!/usr/bin/env python3
"""收集并整理 PostOCR 的有效结果

功能：
1. 遍历 postocr 目录，按书收集所有 accepted 字符
2. 生成统计报告（JSON + 文本）
3. 可选：复制字符图片到按书整理的目录
"""
from __future__ import annotations
import os
import sys
import json
import shutil
from collections import defaultdict
from typing import Dict, List, Any
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src.config import POSTOCR_DIR, ANALYSIS_DIR
except Exception as e:
    raise RuntimeError(f"无法导入配置: {e}")


def collect_book_results(book_dir: str, book_name: str) -> Dict[str, Any]:
    """收集单本书的有效结果

    Args:
        book_dir: 书的目录路径
        book_name: 书名

    Returns:
        {
            'book_name': str,
            'total_chars': int,
            'unique_chars': int,
            'characters': [{'char', 'count', 'files': [...]}, ...],
            'by_version': {version: {...}, ...}
        }
    """
    result = {
        'book_name': book_name,
        'total_chars': 0,
        'unique_chars': 0,
        'characters': [],
        'by_version': {}
    }

    # 字符频次统计
    char_files = defaultdict(list)  # char -> [(version, region, file_path), ...]

    # 遍历所有 preprocessed 版本
    for entry in os.listdir(book_dir):
        version_dir = os.path.join(book_dir, entry)

        # 跳过非目录和特殊目录
        if not os.path.isdir(version_dir):
            continue
        if entry == '_batch_work':
            continue
        if not entry.endswith('_preprocessed'):
            continue

        version_name = entry
        version_stats = {
            'total_chars': 0,
            'unique_chars': 0,
            'regions': []
        }

        # 遍历所有 region
        for region_entry in os.listdir(version_dir):
            region_dir = os.path.join(version_dir, region_entry)

            if not os.path.isdir(region_dir):
                continue
            if not region_entry.startswith('region_'):
                continue

            # 读取 postocr_results.json
            results_file = os.path.join(region_dir, 'postocr_results.json')
            if not os.path.isfile(results_file):
                continue

            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                region_stats = {
                    'region': region_entry,
                    'accepted': 0,
                    'characters': []
                }

                # 提取 accepted 字符
                for entry_data in data.get('results', []):
                    if entry_data.get('status') == 'accepted':
                        char = entry_data.get('ark_evaluation', {}).get('character', '')
                        output_file = entry_data.get('output_file', '')

                        if char and output_file:
                            file_path = os.path.join(region_dir, output_file)

                            # 验证文件存在
                            if os.path.isfile(file_path):
                                char_files[char].append({
                                    'version': version_name,
                                    'region': region_entry,
                                    'file': output_file,
                                    'path': file_path
                                })

                                region_stats['accepted'] += 1
                                region_stats['characters'].append({
                                    'char': char,
                                    'file': output_file
                                })

                if region_stats['accepted'] > 0:
                    version_stats['regions'].append(region_stats)
                    version_stats['total_chars'] += region_stats['accepted']

            except Exception as e:
                print(f"  ⚠ 读取失败: {results_file}: {e}")
                continue

        # 统计该版本的唯一字符数
        version_chars = set()
        for region_stat in version_stats['regions']:
            for char_info in region_stat['characters']:
                version_chars.add(char_info['char'])
        version_stats['unique_chars'] = len(version_chars)

        if version_stats['total_chars'] > 0:
            result['by_version'][version_name] = version_stats

    # 汇总统计
    result['total_chars'] = sum(len(files) for files in char_files.values())
    result['unique_chars'] = len(char_files)

    # 按字符频次排序
    result['characters'] = [
        {
            'char': char,
            'count': len(files),
            'occurrences': files
        }
        for char, files in sorted(char_files.items(), key=lambda x: -len(x[1]))
    ]

    return result


def collect_all_results(postocr_dir: str) -> Dict[str, Any]:
    """收集所有书的有效结果

    Returns:
        {
            'summary': {...},
            'books': [{...}, ...]
        }
    """
    books_data = []

    print("\n=== 收集 PostOCR 有效结果 ===")
    print(f"PostOCR 目录: {postocr_dir}\n")

    # 遍历所有书
    for book_name in sorted(os.listdir(postocr_dir)):
        book_dir = os.path.join(postocr_dir, book_name)

        if not os.path.isdir(book_dir):
            continue
        if book_name.startswith('.'):
            continue
        if book_name.startswith('_'):
            continue

        print(f"[处理] {book_name}")

        book_result = collect_book_results(book_dir, book_name)

        if book_result['total_chars'] > 0:
            books_data.append(book_result)
            print(f"  ✓ 总字符: {book_result['total_chars']}, 去重: {book_result['unique_chars']}")
        else:
            print(f"  - 无有效字符")

    # 生成汇总统计
    summary = {
        'total_books': len(books_data),
        'total_chars': sum(book['total_chars'] for book in books_data),
        'total_unique_chars': len(set(
            char_info['char']
            for book in books_data
            for char_info in book['characters']
        )),
        'books': [
            {
                'name': book['book_name'],
                'total_chars': book['total_chars'],
                'unique_chars': book['unique_chars']
            }
            for book in books_data
        ]
    }

    return {
        'summary': summary,
        'books': books_data
    }


def save_results(data: Dict[str, Any], output_dir: str, copy_files: bool = False):
    """保存收集结果

    Args:
        data: 收集的数据
        output_dir: 输出目录
        copy_files: 是否复制字符图片
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存完整 JSON
    json_path = os.path.join(output_dir, 'collected_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 完整结果已保存: {json_path}")

    # 2. 保存汇总统计
    summary = data['summary']
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PostOCR 结果汇总\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总书籍数: {summary['total_books']}\n")
        f.write(f"总字符数: {summary['total_chars']}\n")
        f.write(f"去重字符数: {summary['total_unique_chars']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("各书统计:\n")
        f.write("-" * 60 + "\n")
        for book_info in summary['books']:
            f.write(f"\n{book_info['name']}\n")
            f.write(f"  总字符: {book_info['total_chars']}\n")
            f.write(f"  去重: {book_info['unique_chars']}\n")

    print(f"✓ 汇总统计已保存: {summary_path}")

    # 3. 保存字符频次统计
    char_freq_path = os.path.join(output_dir, 'character_frequency.txt')
    all_chars = defaultdict(int)
    for book in data['books']:
        for char_info in book['characters']:
            all_chars[char_info['char']] += char_info['count']

    with open(char_freq_path, 'w', encoding='utf-8') as f:
        f.write("字符频次统计（全库去重）\n")
        f.write("=" * 60 + "\n\n")
        for char, count in sorted(all_chars.items(), key=lambda x: -x[1]):
            f.write(f"{char}: {count}\n")

    print(f"✓ 字符频次已保存: {char_freq_path}")

    # 4. 按书保存字符列表
    for book in data['books']:
        book_name = book['book_name']
        safe_book_name = book_name.replace('/', '_').replace(' ', '_')

        book_chars_path = os.path.join(output_dir, f'{safe_book_name}_chars.txt')
        with open(book_chars_path, 'w', encoding='utf-8') as f:
            f.write(f"{book_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总字符: {book['total_chars']}\n")
            f.write(f"去重: {book['unique_chars']}\n\n")

            f.write("字符列表（按频次排序）:\n")
            f.write("-" * 60 + "\n")
            for char_info in book['characters']:
                f.write(f"{char_info['char']}: {char_info['count']} 次\n")

            f.write("\n详细出现位置:\n")
            f.write("-" * 60 + "\n")
            for char_info in book['characters']:
                f.write(f"\n{char_info['char']} ({char_info['count']} 次):\n")
                for occ in char_info['occurrences']:
                    f.write(f"  - {occ['version']} / {occ['region']} / {occ['file']}\n")

    print(f"✓ 各书字符列表已保存到 {output_dir}/")

    # 5. 可选：复制文件
    if copy_files:
        print("\n[复制字符文件...]")
        for book in data['books']:
            book_name = book['book_name']
            safe_book_name = book_name.replace('/', '_').replace(' ', '_')

            book_chars_dir = os.path.join(output_dir, 'characters', safe_book_name)
            os.makedirs(book_chars_dir, exist_ok=True)

            for idx, char_info in enumerate(book['characters'], 1):
                char = char_info['char']

                # 为每个字符创建子目录
                char_dir = os.path.join(book_chars_dir, f"{idx:04d}_{char}")
                os.makedirs(char_dir, exist_ok=True)

                # 复制所有出现的字符图片
                for occ_idx, occ in enumerate(char_info['occurrences'], 1):
                    src_path = occ['path']
                    dst_name = f"{occ_idx:03d}_{occ['version']}_{occ['region']}.png"
                    dst_path = os.path.join(char_dir, dst_name)

                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        print(f"  ⚠ 复制失败: {src_path}: {e}")

            print(f"  ✓ {book_name}: {book['unique_chars']} 个字符")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='收集 PostOCR 有效结果')
    parser.add_argument('--input', default=None, help='PostOCR 输入目录（默认使用配置）')
    parser.add_argument('--output', default=None, help='输出目录（默认使用 ANALYSIS_DIR）')
    parser.add_argument('--copy-files', action='store_true', help='复制字符图片到整理目录')
    args = parser.parse_args()

    input_dir = args.input or POSTOCR_DIR
    output_dir = args.output or os.path.join(ANALYSIS_DIR, 'collected')

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 收集结果
    data = collect_all_results(input_dir)

    # 保存结果
    print("\n=== 保存结果 ===")
    save_results(data, output_dir, copy_files=args.copy_files)

    # 打印汇总
    summary = data['summary']
    print("\n" + "=" * 60)
    print("收集完成！")
    print("=" * 60)
    print(f"总书籍数: {summary['total_books']}")
    print(f"总字符数: {summary['total_chars']}")
    print(f"去重字符数: {summary['total_unique_chars']}")
    print(f"\n结果目录: {output_dir}")


if __name__ == '__main__':
    main()
