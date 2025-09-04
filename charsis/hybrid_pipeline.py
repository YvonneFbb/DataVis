#!/usr/bin/env python3
"""
混合处理流程主脚本
Pipeline: Preprocess → OCR Region Detection → Character Segmentation
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess.core import preprocess_image
from ocr.remote_paddleocr import RemotePaddleOCRClient
from utils.path import ensure_dir

def create_pipeline_structure(base_name: str) -> Dict[str, Path]:
    """创建流水线目录结构"""
    base_path = Path("data/results")
    
    structure = {
        'preprocessed': base_path / "preprocessed" / base_name,
        'ocr': base_path / "ocr" / base_name,
        'segments': base_path / "segments" / base_name,
        'ocr_regions': base_path / "ocr" / base_name / "region_images",
        'segments_summary': base_path / "segments" / base_name / "all_characters"
    }
    
    # 创建所有目录
    for path in structure.values():
        ensure_dir(str(path))
    
    return structure

def step1_preprocess(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    """步骤1: 图像预处理"""
    print(f"\n{'='*50}")
    print(f"步骤1: 图像预处理")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # 保存预处理结果路径
        enhanced_path = output_dir / "enhanced.jpg"
        
        # 执行预处理
        success = preprocess_image(str(input_path), str(enhanced_path))
        
        if not success:
            return {'success': False, 'error': '预处理失败'}
        
        # 读取处理后的图像获取尺寸信息
        import cv2
        enhanced_image = cv2.imread(str(enhanced_path))
        
        # 保存元数据
        metadata = {
            'step': 'preprocess',
            'input_image': str(input_path),
            'output_image': str(enhanced_path),
            'process_time': time.time() - start_time,
            'image_shape': enhanced_image.shape,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 预处理完成: {enhanced_path}")
        print(f"   图像尺寸: {enhanced_image.shape}")
        print(f"   处理耗时: {metadata['process_time']:.2f}秒")
        
        return {
            'success': True,
            'enhanced_path': enhanced_path,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        return {'success': False, 'error': str(e)}

def step2_ocr_regions(input_path: Path, output_dir: Path, regions_dir: Path) -> Dict[str, Any]:
    """步骤2: OCR区域检测"""
    print(f"\n{'='*50}")
    print(f"步骤2: OCR区域检测")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # 初始化PaddleOCR客户端
        client = RemotePaddleOCRClient()
        
        # 执行OCR识别
        result = client.predict_full_document(input_path)
        
        if not result['success']:
            return {'success': False, 'error': result.get('message', '未知错误')}
        
        # 读取原图用于区域切割
        import cv2
        original_image = cv2.imread(str(input_path))
        
        # 保存每个区域为独立图片
        region_info_list = []
        text_regions = result['text_regions']
        
        for i, region in enumerate(text_regions):
            region_id = f"region_{i+1:03d}"
            bbox = region['bbox']
            
            # 提取区域图像
            if isinstance(bbox[0], (list, tuple)):
                # 四点坐标格式，计算边界矩形
                points = [[int(pt[0]), int(pt[1])] for pt in bbox]
                x_coords = [pt[0] for pt in points]
                y_coords = [pt[1] for pt in points]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
            else:
                # 矩形格式
                x1, y1, x2, y2 = bbox[:4]
            
            # 确保坐标在图像范围内
            h, w = original_image.shape[:2]
            x1 = max(0, min(int(x1), w-1))
            y1 = max(0, min(int(y1), h-1))
            x2 = max(x1+1, min(int(x2), w))
            y2 = max(y1+1, min(int(y2), h))
            
            # 提取区域图像
            region_image = original_image[y1:y2, x1:x2]
            
            # 保存区域图像
            region_path = regions_dir / f"{region_id}.jpg"
            cv2.imwrite(str(region_path), region_image)
            
            # 记录区域信息
            region_info = {
                'id': region_id,
                'bbox': bbox,
                'rect_bbox': [x1, y1, x2, y2],  # 标准矩形坐标
                'confidence': region['confidence'],
                'text': region['text'],
                'image_path': f"region_images/{region_id}.jpg",
                'image_size': [x2-x1, y2-y1]
            }
            region_info_list.append(region_info)
            
            print(f"   区域 {i+1:2d}: {region['text'][:20]}... (置信度: {region['confidence']:.3f})")
        
        # 保存区域信息JSON
        regions_data = {
            'total_regions': len(text_regions),
            'regions': region_info_list,
            'ocr_metadata': {
                'server_time': result.get('server_time', 0),
                'total_characters': sum(len(r['text']) for r in text_regions),
                'avg_confidence': sum(r['confidence'] for r in text_regions) / len(text_regions)
            }
        }
        
        with open(output_dir / "regions.json", 'w', encoding='utf-8') as f:
            json.dump(regions_data, f, ensure_ascii=False, indent=2)
        
        # 创建标注图像
        annotated_image = original_image.copy()
        from PIL import Image, ImageDraw, ImageFont
        
        pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        
        for i, region_info in enumerate(region_info_list):
            color = colors[i % len(colors)]
            bbox = region_info['rect_bbox']
            
            # 绘制边界框
            draw.rectangle(bbox, outline=color, width=3)
            
            # 绘制区域ID
            draw.text((bbox[0], bbox[1]-25), f"R{i+1}", fill=color)
        
        # 保存标注图像
        annotated_path = output_dir / "annotated.jpg"
        annotated_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(annotated_path), annotated_cv)
        
        total_time = time.time() - start_time
        
        print(f"✅ OCR区域检测完成")
        print(f"   检测到区域: {len(text_regions)}个")
        print(f"   平均置信度: {regions_data['ocr_metadata']['avg_confidence']:.3f}")
        print(f"   总字符数: {regions_data['ocr_metadata']['total_characters']}")
        print(f"   处理耗时: {total_time:.2f}秒")
        
        return {
            'success': True,
            'regions_data': regions_data,
            'regions_dir': regions_dir,
            'process_time': total_time
        }
        
    except Exception as e:
        print(f"❌ OCR区域检测失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def step3_character_segmentation(regions_data: Dict[str, Any], regions_dir: Path, 
                                output_dir: Path, summary_dir: Path) -> Dict[str, Any]:
    """步骤3: 字符分割（基于OCR区域）"""
    print(f"\n{'='*50}")
    print(f"步骤3: 字符分割")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        from segmentation.vertical_hybrid_segmentation import segment_vertical_column_hybrid
        from segmentation.morphology_segmentation import segment_vertical_column_morphology
        
        total_characters = 0
        all_character_info = []
        region_results = []
        
        # 处理每个区域
        for region_info in regions_data['regions']:
            region_id = region_info['id']
            region_path = regions_dir / f"{region_id}.jpg"
            region_output_dir = output_dir / region_id
            ensure_dir(str(region_output_dir))
            
            print(f"\n处理区域 {region_id}: {region_info['text'][:30]}...")
            
            # 优先使用Hybrid切割；失败则回退到形态学
            segment_result = segment_vertical_column_hybrid(
                str(region_path),
                str(region_output_dir),
                region_info,
                debug=True
            )
            if not (segment_result and segment_result.get('success') and segment_result.get('character_count', 0) > 0):
                print("   Hybrid切割结果不理想，回退到形态学方法…")
                segment_result = segment_vertical_column_morphology(
                    str(region_path),
                    str(region_output_dir),
                    region_info
                )
            
            # segment_vertical_column 返回 dict 格式
            if segment_result and segment_result.get('success'):
                char_count = segment_result['character_count']
                total_characters += char_count
                
                # 从分割结果中获取字符信息
                region_chars = []
                for i, char_data in enumerate(segment_result['characters']):
                    char_info = {
                        'char_id': f"{region_id}_char_{i+1:03d}",
                        'local_file': char_data['filename'],
                        'region_id': region_id,
                        'height': char_data['height'],
                        'width': char_data.get('width', 0),
                        'original_bbox': char_data.get('original_bbox', (0, 0, 0, 0)),
                        'refined_bbox': char_data.get('refined_bbox', (0, 0, 0, 0))
                    }
                    region_chars.append(char_info)
                
                all_character_info.extend(region_chars)
                region_results.append({
                    'region_id': region_id,
                    'character_count': char_count,
                    'characters': region_chars,
                    'ocr_text': region_info['text'],
                    'confidence': region_info['confidence'],
                    'method': segment_result.get('method', 'morphology_only'),
                    'refinement_stats': segment_result.get('refinement_stats', {})
                })
                
                method_info = segment_result.get('method', 'unknown')
                refinement_info = ""
                if 'refinement_stats' in segment_result:
                    stats = segment_result['refinement_stats']
                    boundary_adj = stats.get('boundary_adjustments', 0)
                    size_red = stats.get('size_reductions', 0)
                    refinement_info = f", 边界调整: {boundary_adj}个, 尺寸优化: {size_red}个"
                
                print(f"   ✅ 分割完成: {char_count}个字符 (方法: {method_info}{refinement_info})")
            else:
                error_msg = segment_result.get('error', '分割失败') if segment_result else '分割函数返回空结果'
                print(f"   ❌ 分割失败: {error_msg}")
                region_results.append({
                    'region_id': region_id,
                    'character_count': 0,
                    'error': error_msg
                })
        
        # 创建全局字符图像汇总
        create_global_character_summary(all_character_info, summary_dir, regions_dir)
        
        # 保存汇总统计
        summary_data = {
            'pipeline_info': {
                'step': 'character_segmentation',
                'total_regions_processed': len([r for r in region_results if 'error' not in r]),
                'total_characters': total_characters,
                'process_time': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'region_results': region_results,
            'character_summary': {
                'total_count': len(all_character_info),
                'avg_per_region': len(all_character_info) / len([r for r in region_results if 'error' not in r]) if any('error' not in r for r in region_results) else 0
            }
        }
        
        with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 字符分割完成")
        print(f"   处理区域: {summary_data['pipeline_info']['total_regions_processed']}个")
        print(f"   总字符数: {total_characters}")
        print(f"   处理耗时: {summary_data['pipeline_info']['process_time']:.2f}秒")
        
        return {
            'success': True,
            'summary_data': summary_data,
            'total_characters': total_characters
        }
        
    except Exception as e:
        print(f"❌ 字符分割失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def convert_to_global_coordinates(local_chars: List[Dict], region_bbox: List[int]) -> List[Dict]:
    """将局部坐标转换为全局坐标"""
    x_offset, y_offset = region_bbox[0], region_bbox[1]
    
    global_chars = []
    for char in local_chars:
        global_char = char.copy()
        # 转换边界框坐标
        if 'bbox' in char:
            local_bbox = char['bbox']
            global_bbox = [
                local_bbox[0] + x_offset,
                local_bbox[1] + y_offset,
                local_bbox[2] + x_offset,
                local_bbox[3] + y_offset
            ]
            global_char['bbox'] = global_bbox
        
        global_chars.append(global_char)
    
    return global_chars

def create_global_character_summary(all_chars: List[Dict], summary_dir: Path, regions_dir: Path) -> None:
    """创建全局字符汇总"""
    import shutil
    
    # 将所有字符图像复制到汇总目录并重新编号
    for i, char_info in enumerate(all_chars):
        region_id = char_info['region_id']
        local_file = char_info['local_file']
        
        # 源文件路径
        source_path = regions_dir.parent / "segments" / regions_dir.name / region_id / local_file
        
        # 目标文件路径（全局编号）
        global_filename = f"char_{i+1:04d}.jpg"
        target_path = summary_dir / global_filename
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            char_info['global_file'] = global_filename

def run_hybrid_pipeline(input_image: str = "data/raw/demo.jpg", 
                        skip_preprocess: bool = False,
                        skip_ocr: bool = False,
                        skip_segment: bool = False) -> Dict[str, Any]:
    """运行完整的混合处理流程"""
    print(f"\n🚀 开始混合处理流程")
    print(f"输入图像: {input_image}")
    print(f"跳过步骤: {'预处理' if skip_preprocess else ''}{'OCR' if skip_ocr else ''}{'分割' if skip_segment else ''}")
    
    input_path = Path(input_image)
    if not input_path.exists():
        return {'success': False, 'error': f'输入文件不存在: {input_image}'}
    
    # 创建目录结构
    base_name = input_path.stem
    dirs = create_pipeline_structure(base_name)
    
    pipeline_start = time.time()
    results = {}
    
    # 步骤1: 预处理
    if not skip_preprocess:
        step1_result = step1_preprocess(input_path, dirs['preprocessed'])
        results['step1'] = step1_result
        
        if not step1_result['success']:
            return {'success': False, 'failed_at': 'step1', 'results': results}
        enhanced_path = step1_result['enhanced_path']
    else:
        print(f"\n⏭️ 跳过步骤1: 预处理")
        # 查找现有的预处理结果
        enhanced_path = dirs['preprocessed'] / "enhanced.jpg"
        if not enhanced_path.exists():
            print(f"警告: 未找到预处理结果 {enhanced_path}，使用原始图像")
            enhanced_path = input_path
        results['step1'] = {'success': True, 'skipped': True}
    
    # 步骤2: OCR区域检测
    if not skip_ocr:
        step2_result = step2_ocr_regions(enhanced_path, dirs['ocr'], dirs['ocr_regions'])
        results['step2'] = step2_result
        
        if not step2_result['success']:
            return {'success': False, 'failed_at': 'step2', 'results': results}
    else:
        print(f"\n⏭️ 跳过步骤2: OCR区域检测")
        # 加载现有的OCR结果
        regions_json_path = dirs['ocr'] / "regions.json"
        if not regions_json_path.exists():
            return {'success': False, 'failed_at': 'step2', 'error': f'未找到OCR结果文件: {regions_json_path}'}
        
        with open(regions_json_path, 'r', encoding='utf-8') as f:
            regions_data = json.load(f)
        
        step2_result = {
            'success': True,
            'regions_data': regions_data,
            'regions_dir': dirs['ocr_regions'],
            'skipped': True
        }
        results['step2'] = step2_result
    
    # 步骤3: 字符分割
    if not skip_segment:
        step3_result = step3_character_segmentation(
            step2_result['regions_data'],
            dirs['ocr_regions'],
            dirs['segments'],
            dirs['segments_summary']
        )
        results['step3'] = step3_result
        
        if not step3_result['success']:
            return {'success': False, 'failed_at': 'step3', 'results': results}
    else:
        print(f"\n⏭️ 跳过步骤3: 字符分割")
        results['step3'] = {'success': True, 'skipped': True, 'total_characters': 0}
    
    # 汇总结果
    total_time = time.time() - pipeline_start
    
    final_result = {
        'success': True,
        'pipeline_time': total_time,
        'results': results,
        'summary': {
            'total_regions': step2_result['regions_data']['total_regions'],
            'total_characters': step3_result['total_characters'],
            'avg_chars_per_region': step3_result['total_characters'] / step2_result['regions_data']['total_regions']
        }
    }
    
    # 保存流程汇总 - 确保Path对象转换为字符串
    def convert_paths_to_str(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths_to_str(v) for v in obj]
        return obj
    
    serializable_result = convert_paths_to_str(final_result)
    with open(dirs['segments'] / "pipeline_summary.json", 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 混合处理流程完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"检测区域: {final_result['summary']['total_regions']}个")
    print(f"分割字符: {final_result['summary']['total_characters']}个")
    print(f"平均每区域: {final_result['summary']['avg_chars_per_region']:.1f}个字符")
    
    return final_result

if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='混合处理流程')
    parser.add_argument('--input', '-i', default='data/raw/demo.jpg', help='输入图像路径')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳过预处理步骤')
    parser.add_argument('--skip-ocr', action='store_true', help='跳过OCR步骤')
    parser.add_argument('--skip-segment', action='store_true', help='跳过分割步骤')
    
    args = parser.parse_args()
    
    result = run_hybrid_pipeline(
        input_image=args.input,
        skip_preprocess=args.skip_preprocess,
        skip_ocr=args.skip_ocr,
        skip_segment=args.skip_segment
    )
    
    if result['success']:
        print(f"\n✅ 流程执行成功！")
    else:
        print(f"\n❌ 流程执行失败于: {result.get('failed_at', '未知阶段')}")