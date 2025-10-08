#!/usr/bin/env python3
"""
简化管线脚本：自动识别输入(文件或目录)，按阶段串行执行。

阶段关键字 (按默认顺序): preprocess -> preocr -> segment -> ocr

用法示例:
1) 对单张原始图跑全流程 (所有阶段):
   python -m src.pipeline input.jpg
2) 仅做预处理 + 分割 (跳过 preocr / ocr):
   python -m src.pipeline input.jpg --stages preprocess,segment
3) 对一个目录批量处理 (目录中所有支持的图片):
   python -m src.pipeline data/raw --stages preprocess,segment,ocr
4) 已有 preOCR 结果 (有 region_images 子目录)，直接分割 + OCR:
   python -m src.pipeline data/results/preocr/demo --stages segment,ocr

所有参数依赖 config.py，不再暴露细粒度 CLI 选项；如需高级调参使用 pipeline_advanced.py。
"""
from __future__ import annotations
import os, sys, json, glob, shutil
from typing import List, Dict, Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src import config
    from src.config import (
        RAW_DIR, PREPROCESSED_DIR, PREOCR_DIR, SEGMENTS_DIR, POSTOCR_DIR
    )
except Exception as e:
    raise RuntimeError(f"配置导入失败: {e}")

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def _is_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path.lower())[1] in IMG_EXT


def _list_images_in_dir(folder: str) -> List[str]:
    """递归遍历目录查找所有支持的图片文件"""
    files = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in IMG_EXT):
                files.append(os.path.join(root, filename))
    files.sort()
    return files


def stage_preprocess(inputs: List[str], input_base_dir: str) -> List[str]:
    from src.preprocess.core import preprocess_image
    outputs = []
    for p in inputs:
        # 计算相对路径以保持目录结构
        rel_path = os.path.relpath(p, input_base_dir)
        rel_dir = os.path.dirname(rel_path)
        base, ext = os.path.splitext(os.path.basename(p))

        # 创建对应的输出目录结构
        if rel_dir and rel_dir != '.':
            out_dir = os.path.join(PREPROCESSED_DIR, rel_dir)
        else:
            out_dir = PREPROCESSED_DIR
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{base}_preprocessed{ext}")
        ok = preprocess_image(p, out_path)
        if ok:
            outputs.append(out_path)
    return outputs


def stage_preocr(inputs: List[str], input_base_dir: str) -> Dict[str, Any]:
    # 使用远程服务对每张图做区域检测
    from src.preocr.run_preocr import run_preocr_on_single_image
    summary: Dict[str, Any] = {'inputs': len(inputs), 'details': []}
    for p in inputs:
        try:
            # 计算相对路径以保持目录结构
            rel_path = os.path.relpath(p, input_base_dir)
            rel_dir = os.path.dirname(rel_path)
            r = run_preocr_on_single_image(p, rel_dir)
        except Exception as e:
            r = {'error': str(e), 'input': p}
        summary['details'].append(r)
    return summary


def _collect_region_images(preocr_root: str) -> List[str]:
    """
    收集 region_images 目录下的所有图片文件
    支持两种结构:
    1. preocr/region_images/*.jpg
    2. preocr/<dataset>/region_images/*.jpg
    3. preocr/<dataset>/<image>/region_images/*.jpg (多层嵌套)
    """
    region_imgs: List[str] = []
    if not os.path.isdir(preocr_root):
        return region_imgs

    # 直接在根目录下的 region_images
    region_dir = os.path.join(preocr_root, 'region_images')
    if os.path.isdir(region_dir):
        for ext in IMG_EXT:
            region_imgs.extend(glob.glob(os.path.join(region_dir, f'*{ext}')))
    else:
        # 递归查找所有 region_images 目录
        for root, dirs, _ in os.walk(preocr_root):
            if 'region_images' in dirs:
                region_dir = os.path.join(root, 'region_images')
                for ext in IMG_EXT:
                    region_imgs.extend(glob.glob(os.path.join(region_dir, f'*{ext}')))

    region_imgs.sort()
    return region_imgs


def stage_segment(region_images: List[str]) -> Dict[str, Any]:
    from src.segmentation.vertical_hybrid import run_on_image
    processed = []
    for img_path in region_images:
        # out dir: segments/<dataset_or_misc>/<subdirs...>/<region_name>
        # 解析路径: 如果路径包含 /preocr/<ds>/.../region_images/region_xxx.jpg
        parts = img_path.split(os.sep)
        try:
            idx = parts.index('preocr')
            # 找到 region_images 的位置
            try:
                region_idx = parts.index('region_images', idx)
                # 提取 preocr 之后到 region_images 之前的所有目录（包括 dataset 和子目录）
                if region_idx > idx + 1:
                    dataset_path = os.path.join(*parts[idx+1:region_idx])
                else:
                    dataset_path = parts[idx+1] if idx+1 < len(parts) else 'misc'
            except ValueError:
                # 没有 region_images 目录，只取 dataset
                dataset_path = parts[idx+1] if idx+1 < len(parts) else 'misc'
        except ValueError:
            dataset_path = 'misc'
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(SEGMENTS_DIR, dataset_path, base)
        os.makedirs(out_dir, exist_ok=True)
        res = run_on_image(img_path, out_dir, framework='livetext')
        overlay_path = res.get('overlay')
        overlay_target = None
        if overlay_path and os.path.isfile(overlay_path):
            overlay_dir = os.path.join(SEGMENTS_DIR, dataset_path, '_overlays')
            os.makedirs(overlay_dir, exist_ok=True)
            overlay_target = os.path.join(overlay_dir, f"{base}_overlay.png")
            try:
                shutil.copy2(overlay_path, overlay_target)
            except Exception as e:
                print(f"warning: overlay copy failed for {overlay_path}: {e}")
                overlay_target = None
        processed.append({
            'image': img_path,
            'out_dir': out_dir,
            'count': res.get('character_count', 0),
            'overlay': overlay_path,
            'overlay_aggregated': overlay_target,
        })
    return {'processed': processed}


def stage_postocr(input_dir: str = None) -> Dict[str, Any]:
    from src.postocr.core import process_all_segment_results
    force = bool(os.getenv('POSTOCR_FORCE'))
    return process_all_segment_results(input_base_dir=input_dir, config={'force': force})


def run_pipeline(input_path: str, stages: List[str]) -> Dict[str, Any]:
    config.validate_config()
    result: Dict[str, Any] = {'input': input_path, 'stages': stages}
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'输入不存在: {input_path}')

    # 1. 收集初始图片列表 (raw set)
    if os.path.isdir(input_path):
        raw_images = _list_images_in_dir(input_path)
    elif _is_image(input_path):
        raw_images = [input_path]
    else:
        raise ValueError('输入既不是图片也不是目录')
    result['raw_count'] = len(raw_images)

    current_images = raw_images
    region_images: List[str] = []

    for st in stages:
        if st == 'preprocess':
            current_images = stage_preprocess(current_images, input_path)
            result['preprocess'] = {'count': len(current_images)}
        elif st == 'preocr':
            # preocr 阶段使用 preprocessed 目录作为基准
            preocr_info = stage_preocr(current_images, PREPROCESSED_DIR)
            result['preocr'] = preocr_info
            # 尝试收集 region 图（如果用户已事先跑过 preOCR）
            region_images = _collect_region_images(PREOCR_DIR)
            result['region_images_found'] = len(region_images)
        elif st == 'segment':
            if not region_images:
                # 如果用户直接传入了 preocr 数据集目录，则从该目录收集 region_images
                if os.path.isdir(input_path) and ('preocr' in os.path.normpath(input_path).split(os.sep)):
                    region_images = _collect_region_images(input_path)
                # 否则退回到当前图片列表
                if not region_images:
                    region_images = current_images
            seg_info = stage_segment(region_images)
            result['segment'] = {'processed': len(seg_info.get('processed', []))}
        elif st in ('ocr', 'postocr'):
            post_stats = stage_postocr()
            result['postocr'] = post_stats
        else:
            raise ValueError(f'未知阶段: {st}')

    return result


# ==================== 批量任务管理 ====================
def _build_batch_client():
    """构建批量推理客户端"""
    from src.postocr.batch_client import BatchClient, BatchConfig
    if not config.POSTOCR_CONFIG.get('enabled', False):
        print("PostOCR 未启用")
        return None
    provider = str(config.POSTOCR_CONFIG.get('provider', 'qwen')).lower()
    provider_cfg = config.POSTOCR_CONFIG.get(provider, {})
    if not provider_cfg:
        print(f"未找到 provider '{provider}' 的配置")
        return None
    batch_cfg_dict = config.POSTOCR_CONFIG.get('batch', {})
    return BatchClient(BatchConfig(
        base_url=str(provider_cfg.get('base_url', '')),
        api_key_env=str(provider_cfg.get('api_key_env', 'DASHSCOPE_API_KEY')),
        completion_window=str(batch_cfg_dict.get('completion_window', '24h')),
        poll_interval=int(batch_cfg_dict.get('poll_interval', 60)),
        max_requests_per_batch=int(batch_cfg_dict.get('max_requests_per_batch', 50000)),
        max_line_size_mb=float(batch_cfg_dict.get('max_line_size_mb', 6.0)),
    ))


def cmd_batch_status(batch_id: str):
    """查询批量任务状态"""
    client = _build_batch_client()
    if not client:
        return
    try:
        status_info = client.get_status(batch_id)
        print("\n=== 批量任务状态 ===")
        print(f"任务 ID: {status_info['id']}")
        print(f"状态: {status_info['status']}")
        counts = status_info['request_counts']
        print(f"\n进度:")
        print(f"  总请求数: {counts['total']}")
        print(f"  已完成: {counts['completed']}")
        print(f"  失败: {counts['failed']}")
        if counts['total'] > 0:
            print(f"  完成率: {(counts['completed'] / counts['total']) * 100:.1f}%")
        print(f"\n输出文件 ID: {status_info['output_file_id'] or 'N/A'}")
        print(f"错误文件 ID: {status_info['error_file_id'] or 'N/A'}")
    except Exception as e:
        print(f"查询失败: {e}")


def cmd_batch_resume(work_dir: str):
    """从断点恢复批量任务"""
    client = _build_batch_client()
    if not client:
        return
    if not os.path.isdir(work_dir):
        print(f"工作目录不存在: {work_dir}")
        return
    print(f"\n从工作目录恢复任务: {work_dir}")
    try:
        result = client.resume_batch_inference(work_dir)
        print("\n=== 任务恢复完成 ===")
        print(f"任务 ID: {result['batch_id']}")
        print(f"最终状态: {result['status']}")
        counts = result['request_counts']
        print(f"\n结果统计:")
        print(f"  总请求数: {counts['total']}")
        print(f"  已完成: {counts['completed']}")
        print(f"  失败: {counts['failed']}")
        print(f"\n结果文件: {result['result_file']}")
        print(f"结果数量: {len(result['results'])} 条")
    except Exception as e:
        print(f"恢复失败: {e}")
        import traceback
        traceback.print_exc()


def cmd_batch_download(batch_id: str, output_dir: str = None, wait: bool = False):
    """从 batch_id 下载结果"""
    client = _build_batch_client()
    if not client:
        return
    print(f"\n查询任务状态: {batch_id}")
    try:
        status_info = client.get_status(batch_id)
        print(f"状态: {status_info['status']}")
        if status_info['status'] not in ['completed', 'failed', 'expired', 'cancelled']:
            print(f"任务尚未完成，当前状态: {status_info['status']}")
            if wait:
                print(f"开始等待任务完成...")
                status_info = client.wait_for_completion(batch_id)
                print(f"任务完成，最终状态: {status_info['status']}")
            else:
                print("使用 --wait 参数可等待任务完成")
                return
        output_dir = output_dir or f'./batch_download_{batch_id}'
        os.makedirs(output_dir, exist_ok=True)
        if status_info['output_file_id']:
            result_file = os.path.join(output_dir, 'batch_results.jsonl')
            client.download_results(status_info['output_file_id'], result_file)
            print(f"\n✓ 结果已保存: {result_file}")
            results = client.parse_results(result_file)
            print(f"  共 {len(results)} 条结果")
        else:
            print("没有输出文件")
        if status_info['error_file_id']:
            error_file = os.path.join(output_dir, 'batch_errors.jsonl')
            client.download_results(status_info['error_file_id'], error_file)
            print(f"✓ 错误文件已保存: {error_file}")
        status_file = os.path.join(output_dir, 'batch_status.json')
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_info, f, ensure_ascii=False, indent=2)
        print(f"✓ 状态信息已保存: {status_file}")
    except Exception as e:
        print(f"下载失败: {e}")


def cmd_batch_list(limit: int = 10, local: bool = False):
    """列出所有批量任务"""
    if local:
        # 只显示本地记录的任务
        tasks_file = os.path.join(config.POSTOCR_DIR, '_batch_tasks.json')
        if not os.path.exists(tasks_file):
            print("没有找到本地任务记录")
            return

        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tasks = data.get('tasks', [])
            if not tasks:
                print("没有本地任务记录")
                return

            print("\n=== 本地批量任务列表 ===")
            print(f"总共 {len(tasks)} 个任务\n")

            client = _build_batch_client()

            for task in tasks:
                image_name = task.get('image_name', 'unknown')
                batch_id = task.get('batch_id', 'unknown')
                total_files = task.get('total_files', 0)

                print(f"任务: {image_name}")
                print(f"  batch_id: {batch_id}")
                print(f"  文件数: {total_files}")

                # 尝试查询云端状态
                if client:
                    try:
                        status_info = client.get_status(batch_id)
                        print(f"  状态: {status_info['status']}")
                        counts = status_info['request_counts']
                        print(f"  进度: {counts['completed']}/{counts['total']} (失败: {counts['failed']})")
                    except Exception:
                        print(f"  状态: 查询失败")
                print()
        except Exception as e:
            print(f"读取本地任务失败: {e}")
    else:
        # 显示云端所有任务
        client = _build_batch_client()
        if not client:
            return
        try:
            print("\n=== 云端批量任务列表 ===")
            batches = client.client.batches.list(limit=limit)
            if not batches.data:
                print("没有找到批量任务")
                return
            for batch in batches.data:
                print(f"\n任务 ID: {batch.id}")
                print(f"  状态: {batch.status}")
                print(f"  模型端点: {batch.endpoint}")
                counts = batch.request_counts
                print(f"  进度: {counts.completed}/{counts.total} (失败: {counts.failed})")
                if batch.metadata and batch.metadata.get('ds_name'):
                    print(f"  名称: {batch.metadata.get('ds_name')}")
                print(f"  创建时间: {batch.created_at}")
            print(f"\n显示前 {len(batches.data)} 个任务")
            if batches.has_more:
                print(f"还有更多任务，使用 --limit 增加显示数量")
        except Exception as e:
            print(f"列表查询失败: {e}")


def cmd_batch_wait(output_dir: str):
    """等待所有批量任务完成并下载结果"""
    from src.postocr.core import wait_and_process_batch_tasks
    try:
        result = wait_and_process_batch_tasks(output_dir)
        print("\n=== 批量任务处理完成 ===")
        print(f"成功处理: {result.get('success_count', 0)}/{result.get('total_tasks', 0)} 个任务")
        print(f"总文件数: {result.get('total_files', 0)}")
        print(f"通过过滤: {result.get('total_accepted', 0)}")
        print(f"过滤淘汰: {result.get('total_filtered', 0)}")
    except Exception as e:
        print(f"批量任务处理失败: {e}")
        import traceback
        traceback.print_exc()


def cmd_batch_retry_failed(output_dir: str, mode: str = 'batch'):
    """重试失败的批量任务"""
    from src.postocr.core import retry_failed_requests
    try:
        result = retry_failed_requests(output_dir, mode=mode)
        print("\n=== 失败请求重试完成 ===")
        print(f"批量失败: {result.get('total_failed', 0)} 个")
        print(f"过滤小图: {result.get('filtered_small', 0)} 个")
        print(f"已被实时重试成功: {result.get('already_success', 0)} 个")
        print(f"实际提交重试: {result.get('retry_submitted', 0)} 个")
        if mode == 'batch':
            print(f"\n新批量任务已创建，运行以下命令等待完成：")
            print(f"  ./pipeline postocr wait {output_dir}")
        else:
            print(f"\nRealtime 模式处理完成")
            print(f"  跳过（已处理）: {result.get('retry_skipped', 0)}")
            print(f"  成功: {result.get('retry_success', 0)}")
            print(f"  失败: {result.get('retry_failed', 0)}")
    except Exception as e:
        print(f"重试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主程序入口"""
    import argparse

    # 预处理命令行参数：如果 postocr 后面跟的是路径而非子命令，自动插入 'run'
    if len(sys.argv) >= 3 and sys.argv[1] == 'postocr':
        next_arg = sys.argv[2]
        # 如果不是已知子命令，且看起来像路径，则插入 'run'
        if next_arg not in ('run', 'status', 'resume', 'download', 'list') and \
           (os.path.exists(next_arg) or '/' in next_arg or os.path.sep in next_arg):
            sys.argv.insert(2, 'run')
    parser = argparse.ArgumentParser(
        description='古籍处理流水线：各阶段独立运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 预处理阶段:
   ./pipeline preprocess input.jpg
   ./pipeline preprocess data/raw/

2. PreOCR 区域检测:
   ./pipeline preocr preprocessed_image.jpg

3. 字符分割:
   ./pipeline segment region_image.jpg

4. PostOCR 质量过滤:
   ./pipeline postocr                    # 处理所有 segments 目录
   ./pipeline postocr status <batch_id>  # 查询批量任务状态
   ./pipeline postocr resume <work_dir>  # 恢复批量任务
   ./pipeline postocr download <batch_id> [--wait]
   ./pipeline postocr list [--limit N]   # 列出所有批量任务

5. 运行完整流水线:
   ./pipeline all input.jpg                        # 全流程
   ./pipeline all data/raw/ --stages segment,postocr  # 指定阶段

注意: 也可使用模块方式: python -m src.pipeline <command>
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='阶段命令')

    # preprocess 命令
    parser_preprocess = subparsers.add_parser('preprocess', help='预处理：去噪、矫正')
    parser_preprocess.add_argument('input', help='输入文件或目录')
    parser_preprocess.add_argument('--json', action='store_true', help='输出 JSON 结果')

    # preocr 命令
    parser_preocr = subparsers.add_parser('preocr', help='PreOCR：区域检测')
    parser_preocr.add_argument('input', help='输入文件或目录')
    parser_preocr.add_argument('--json', action='store_true', help='输出 JSON 结果')

    # segment 命令
    parser_segment = subparsers.add_parser('segment', help='分割：字符切割')
    parser_segment.add_argument('input', help='输入区域图片或目录')
    parser_segment.add_argument('--json', action='store_true', help='输出 JSON 结果')

    # postocr 命令组（包含批量管理）
    parser_postocr = subparsers.add_parser('postocr', help='PostOCR：质量过滤与批量管理')
    postocr_subparsers = parser_postocr.add_subparsers(dest='postocr_command', help='PostOCR 操作')

    # postocr run：运行过滤（默认）
    parser_postocr_run = postocr_subparsers.add_parser('run', help='运行质量过滤（默认）')
    parser_postocr_run.add_argument('input', nargs='?', help='输入目录（可选，默认处理整个 segments 目录）')
    parser_postocr_run.add_argument('--json', action='store_true', help='输出 JSON 结果')

    # postocr status
    parser_postocr_status = postocr_subparsers.add_parser('status', help='查询批量任务状态')
    parser_postocr_status.add_argument('batch_id', help='批量任务 ID')

    # postocr resume
    parser_postocr_resume = postocr_subparsers.add_parser('resume', help='从断点恢复批量任务')
    parser_postocr_resume.add_argument('work_dir', help='批量任务工作目录')

    # postocr download
    parser_postocr_download = postocr_subparsers.add_parser('download', help='下载批量任务结果')
    parser_postocr_download.add_argument('batch_id', help='批量任务 ID')
    parser_postocr_download.add_argument('--output-dir', help='输出目录')
    parser_postocr_download.add_argument('--wait', action='store_true', help='等待任务完成')

    # postocr list
    parser_postocr_list = postocr_subparsers.add_parser('list', help='列出批量任务（默认本地）')
    parser_postocr_list.add_argument('--cloud', action='store_true', help='显示云端所有任务')
    parser_postocr_list.add_argument('--limit', type=int, default=10, help='显示数量（仅云端模式）')

    # postocr wait
    parser_postocr_wait = postocr_subparsers.add_parser('wait', help='等待所有批量任务完成并下载结果')
    parser_postocr_wait.add_argument('output_dir', help='PostOCR输出目录（包含 _batch_tasks.json）')

    # postocr retry-failed
    parser_postocr_retry = postocr_subparsers.add_parser('retry-failed', help='重试失败的批量任务')
    parser_postocr_retry.add_argument('output_dir', help='PostOCR输出目录（包含 _batch_tasks.json）')
    parser_postocr_retry.add_argument('--mode', choices=['batch', 'realtime'], default='batch',
                                      help='重试模式（默认: batch）')

    # all 命令（完整流水线）
    parser_all = subparsers.add_parser('all', help='运行完整流水线')
    parser_all.add_argument('input', help='输入文件或目录')
    parser_all.add_argument('--stages', default='preprocess,segment,postocr', help='逗号分隔阶段序列')
    parser_all.add_argument('--json', action='store_true', help='输出 JSON 结果')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # 处理各个命令
    try:
        if args.command == 'preprocess':
            out = run_pipeline(args.input, ['preprocess'])
            if args.json:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"完成预处理: 处理 {out.get('preprocess', {}).get('count', 0)} 个文件")

        elif args.command == 'preocr':
            out = run_pipeline(args.input, ['preocr'])
            if args.json:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                details = out.get('preocr', {}).get('details', [])
                print(f"完成 PreOCR: 处理 {len(details)} 个文件，发现 {out.get('region_images_found', 0)} 个区域")

        elif args.command == 'segment':
            out = run_pipeline(args.input, ['segment'])
            if args.json:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"完成分割: 处理 {out.get('segment', {}).get('processed', 0)} 个区域")

        elif args.command == 'postocr':
            if not args.postocr_command or args.postocr_command == 'run':
                # 默认：运行 postocr
                input_dir = getattr(args, 'input', None)
                post_stats = stage_postocr(input_dir=input_dir)
                if hasattr(args, 'json') and args.json:
                    print(json.dumps(post_stats, ensure_ascii=False, indent=2))
                else:
                    print(f"完成 PostOCR: 处理 {post_stats.get('processed_dirs', 0)} 个目录")
                    print(f"  接受: {post_stats.get('total_accepted', 0)}, 过滤: {post_stats.get('total_filtered', 0)}")
            elif args.postocr_command == 'status':
                cmd_batch_status(args.batch_id)
            elif args.postocr_command == 'resume':
                cmd_batch_resume(args.work_dir)
            elif args.postocr_command == 'download':
                cmd_batch_download(args.batch_id, args.output_dir, args.wait)
            elif args.postocr_command == 'list':
                cmd_batch_list(args.limit, local=not args.cloud)
            elif args.postocr_command == 'wait':
                cmd_batch_wait(args.output_dir)
            elif args.postocr_command == 'retry-failed':
                cmd_batch_retry_failed(args.output_dir, mode=args.mode)

        elif args.command == 'all':
            stages = [s.strip() for s in args.stages.split(',') if s.strip()]
            out = run_pipeline(args.input, stages)
            if args.json:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"完成流水线: stages={stages}")
                if 'preprocess' in stages:
                    print(f"  预处理: {out.get('preprocess', {}).get('count', 0)} 个文件")
                if 'segment' in stages:
                    print(f"  分割: {out.get('segment', {}).get('processed', 0)} 个区域")
                if 'postocr' in stages or 'ocr' in stages:
                    print(f"  PostOCR: {out.get('postocr', {}).get('processed_dirs', 0)} 个目录")

    except Exception as e:
        print(f"失败: {e}")
        if os.environ.get('PIPELINE_RAISE'):
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
