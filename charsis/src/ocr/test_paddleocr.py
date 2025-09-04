#!/usr/bin/env python3
"""
PaddleOCR 测试脚本
测试PaddleOCR在当前环境下的兼容性和性能
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_paddleocr_installation():
    """测试PaddleOCR安装和导入"""
    try:
        import paddleocr
        print(f"✅ PaddleOCR导入成功，版本: {paddleocr.__version__}")
        return True
    except ImportError as e:
        print(f"❌ PaddleOCR导入失败: {e}")
        print("请先安装PaddleOCR: pip install paddleocr")
        return False

def test_paddleocr_basic():
    """测试PaddleOCR基本功能"""
    try:
        print("\n=== 测试PaddleOCR基本功能 ===")
        
        # 尝试初始化PaddleOCR
        print("正在初始化PaddleOCR (CPU模式)...")
        start_time = time.time()
        
        from paddleocr import PaddleOCR
        
        # CPU模式，使用最简单的配置
        ocr = PaddleOCR(
            use_textline_orientation=False,  # 不使用方向分类器
            lang='ch'            # 中文模式
        )
        
        init_time = time.time() - start_time
        print(f"✅ PaddleOCR初始化成功，耗时: {init_time:.2f}秒")
        
        return ocr
        
    except Exception as e:
        print(f"❌ PaddleOCR初始化失败: {e}")
        return None

def test_on_demo_image(ocr):
    """测试完整的demo.jpg图片"""
    print(f"\n=== 测试完整demo.jpg图片 ===")
    
    # 查找demo.jpg图片
    project_root = Path(__file__).parent.parent.parent
    demo_image = project_root / 'data' / 'raw' / 'demo.jpg'
    
    if not demo_image.exists():
        print(f"❌ demo.jpg不存在: {demo_image}")
        return []
    
    print(f"测试图片: {demo_image}")
    
    try:
        print("正在进行OCR识别...")
        start_time = time.time()
        
        # 进行OCR识别
        result = ocr.predict(input=str(demo_image))
        
        process_time = time.time() - start_time
        print(f"OCR处理完成，耗时: {process_time:.2f}秒")
        
        # 提取识别结果
        all_results = []
        if result and len(result) > 0:
            res = result[0]
            if hasattr(res, 'json') and res.json:
                json_data = res.json['res']
                rec_texts = json_data.get('rec_texts', [])
                rec_scores = json_data.get('rec_scores', [])
                dt_polys = json_data.get('dt_polys', [])
                
                print(f"检测到文本区域数量: {len(dt_polys)}")
                print(f"识别文本数量: {len(rec_texts)}")
                
                # 收集所有识别结果
                for i, text in enumerate(rec_texts):
                    if text and text.strip():  # 只保留非空文本
                        confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                        all_results.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'index': i
                        })
                
                # 按置信度排序
                all_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                print(f"\n识别到的文本内容 (按置信度排序):")
                for i, item in enumerate(all_results[:20]):  # 显示前20个结果
                    print(f"  {i+1:2d}. '{item['text']}' (置信度: {item['confidence']:.3f})")
                
                if len(all_results) > 20:
                    print(f"  ... 还有 {len(all_results)-20} 个结果")
        
        return {
            'total_time': process_time,
            'detected_regions': len(dt_polys) if 'dt_polys' in locals() else 0,
            'recognized_texts': len(all_results),
            'results': all_results,
            'success': len(all_results) > 0
        }
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'total_time': 0.0,
            'detected_regions': 0,
            'recognized_texts': 0,
            'results': [],
            'success': False,
            'error': str(e)
        }

def analyze_demo_results(results: Dict[str, Any]):
    """分析demo图片测试结果"""
    print(f"\n=== demo.jpg测试结果分析 ===")
    
    if not results or not results.get('success'):
        print("❌ 测试失败或无识别结果")
        if results.get('error'):
            print(f"错误信息: {results['error']}")
        return
    
    print(f"✅ 测试成功!")
    print(f"总处理耗时: {results['total_time']:.2f}秒")
    print(f"检测到文本区域: {results['detected_regions']}个")
    print(f"成功识别文本: {results['recognized_texts']}个")
    
    if results['results']:
        # 置信度统计
        confidences = [r['confidence'] for r in results['results']]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"置信度范围: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # 按置信度分类
        high_conf = len([r for r in results['results'] if r['confidence'] >= 0.8])
        medium_conf = len([r for r in results['results'] if 0.5 <= r['confidence'] < 0.8])
        low_conf = len([r for r in results['results'] if r['confidence'] < 0.5])
        
        print(f"\n置信度分布:")
        print(f"  高置信度(≥0.8): {high_conf}个 ({high_conf/len(results['results'])*100:.1f}%)")
        print(f"  中等置信度(0.5-0.8): {medium_conf}个 ({medium_conf/len(results['results'])*100:.1f}%)")
        print(f"  低置信度(<0.5): {low_conf}个 ({low_conf/len(results['results'])*100:.1f}%)")
        
        # 显示一些文本长度统计
        text_lengths = [len(r['text']) for r in results['results']]
        avg_length = sum(text_lengths) / len(text_lengths)
        print(f"\n平均文本长度: {avg_length:.1f}字符")
        
        # 显示常见字符
        all_text = ''.join([r['text'] for r in results['results']])
        unique_chars = set(all_text)
        print(f"识别到的不同字符数: {len(unique_chars)}个")
        
def save_demo_results(results: Dict[str, Any]):
    """保存demo测试结果"""
    if not results:
        return
        
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / 'data' / 'results' / 'paddleocr_demo_results.json'
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加测试元数据
    output_data = {
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_image': 'data/raw/demo.jpg',
        'paddleocr_version': '3.2.0',
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存至: {output_file}")

def main():
    """主函数"""
    print("=== PaddleOCR 兼容性和性能测试 ===")
    
    # 1. 测试安装
    if not test_paddleocr_installation():
        return
    
    # 2. 测试基本功能
    ocr = test_paddleocr_basic()
    if ocr is None:
        return
    
    # 3. 测试完整demo图片
    results = test_on_demo_image(ocr)
    
    # 4. 分析结果
    analyze_demo_results(results)
    
    # 5. 保存结果
    save_demo_results(results)
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    main()