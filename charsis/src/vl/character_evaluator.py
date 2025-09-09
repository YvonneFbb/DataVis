"""
VL字符质量评估模块
使用VL模型评估切割出的字符图片质量，替代传统OCR过滤
"""
import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VL_CONFIG, VL_EVALUATIONS_DIR
from src.utils.path import ensure_dir
from vl.providers import SiliconFlowVision


class VLCharacterEvaluator:
    """
    VL字符质量评估器
    用于判断切割出的字符图片质量
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化字符评估器
        """
        self.config = config or VL_CONFIG.copy()
        
        # 从环境变量获取API key
        api_key = os.environ.get('SILICONFLOW_API_KEY', '')
        if api_key:
            self.config['api_key'] = api_key
        
        # 设置合理的超时时间
        self.config['timeout'] = 60  # 1分钟超时，单字符评估应该很快
        
        self.client = SiliconFlowVision(self.config)
        
        # 字符质量评估提示词
        self.evaluation_prompt = """这是一个从古籍图片中切割出来的字符区域。

请仔细观察并判断：

1. 这是否是一个完整的单个汉字？
2. 切割质量如何？

请按以下格式回答：

如果是完整的单个汉字，回答：GOOD:识别的汉字
如果切割不完整，回答：PARTIAL
如果包含多个字符，回答：MULTIPLE  
如果是噪音或污点，回答：NOISE
如果模糊不清，回答：UNCLEAR

例如：
- GOOD:天
- GOOD:地
- PARTIAL
- MULTIPLE
- NOISE

只需按格式回答，不要添加其他解释。"""
        
        print(f"VL字符质量评估器初始化完成")
        print(f"模型: {self.config['model']}")
    
    def evaluate_character(self, char_image_path: str) -> Dict[str, Any]:
        """
        评估单个字符图片质量
        
        :param char_image_path: 字符图片路径
        :return: 评估结果
        """
        if not os.path.exists(char_image_path):
            return {
                'quality': 'ERROR',
                'error': f'字符图片不存在: {char_image_path}',
                'is_good': False
            }
        
        try:
            # 使用流式识别进行评估
            content, metadata = self.client.recognize_text_stream(
                char_image_path, 
                self.evaluation_prompt, 
                self.config['timeout']
            )
            
            if 'error' in metadata:
                return {
                    'quality': 'ERROR',
                    'error': metadata['error'],
                    'is_good': False
                }
            
            # 解析评估结果
            quality, recognized_char = self._parse_quality(content.strip())
            is_good = quality == 'GOOD'
            
            result = {
                'image_path': char_image_path,
                'quality': quality,
                'recognized_char': recognized_char,
                'raw_response': content.strip(),
                'is_good': is_good,
                'response_time': metadata.get('response_time', 0),
                'model': self.config['model']
            }
            
            return result
            
        except Exception as e:
            return {
                'quality': 'ERROR',
                'error': f'评估失败: {e}',
                'is_good': False
            }
    
    def _parse_quality(self, response: str) -> tuple:
        """
        解析VL模型的质量评估响应
        返回 (quality, recognized_char) 元组
        """
        response = response.strip()
        
        # 处理 GOOD:汉字 格式
        if response.startswith('GOOD:') and len(response) > 5:
            recognized_char = response[5:].strip()
            return 'GOOD', recognized_char
        elif 'GOOD' in response.upper():
            return 'GOOD', None
        elif 'PARTIAL' in response.upper():
            return 'PARTIAL', None
        elif 'MULTIPLE' in response.upper():
            return 'MULTIPLE', None
        elif 'NOISE' in response.upper():
            return 'NOISE', None
        elif 'UNCLEAR' in response.upper():
            return 'UNCLEAR', None
        else:
            # 如果无法识别，记录原始回答
            return f'UNKNOWN({response[:20]})', None
    
    def _sanitize_filename(self, char: str) -> str:
        """
        清理文件名中不合法的字符
        """
        if not char:
            return "unknown"
        
        # 移除可能的文件名不安全字符
        import re
        # 保留中文字符、字母、数字
        safe_char = re.sub(r'[^\u4e00-\u9fff\w]', '', char)
        
        # 如果清理后为空，使用默认名
        if not safe_char:
            return "unknown"
        
        # 限制长度
        return safe_char[:10]
    
    def evaluate_and_filter_characters(self, char_dir: str, vl_output_dir: str = None) -> Dict[str, Any]:
        """
        评估字符质量并将结果复制到VL输出目录
        
        :param char_dir: 字符图片目录（segment结果目录）
        :param vl_output_dir: VL输出目录，如果为None则自动生成
        :return: 评估统计信息
        """
        if not os.path.exists(char_dir):
            print(f"❌ 字符目录不存在: {char_dir}")
            return {'error': '目录不存在'}
        
        # 获取所有字符图片
        char_files = []
        for file in sorted(os.listdir(char_dir)):
            if file.startswith('char_') and file.endswith('.png'):
                char_files.append(file)
        
        if not char_files:
            print(f"❌ 在 {char_dir} 中没有找到字符图片")
            return {'error': '没有字符图片'}
        
        # 自动生成VL输出目录
        if vl_output_dir is None:
            # 从segment目录路径生成对应的VL目录
            segment_name = os.path.basename(char_dir)
            from config import VL_SEGMENTS_DIR
            ensure_dir(VL_SEGMENTS_DIR)
            vl_output_dir = os.path.join(VL_SEGMENTS_DIR, segment_name)
        
        ensure_dir(vl_output_dir)
        
        print(f"\n=== VL字符质量评估与筛选 ===")
        print(f"输入目录: {char_dir}")
        print(f"输出目录: {vl_output_dir}")
        print(f"字符数量: {len(char_files)}")
        
        stats = {
            'total': len(char_files),
            'good': 0,
            'partial': 0, 
            'multiple': 0,
            'noise': 0,
            'unclear': 0,
            'error': 0,
            'processed_files': []
        }
        
        start_time = time.time()
        
        for i, filename in enumerate(char_files, 1):
            old_path = os.path.join(char_dir, filename)
            print(f"[{i}/{len(char_files)}] 处理: {filename}")
            
            # VL评估
            result = self.evaluate_character(old_path)
            quality = result['quality']
            recognized_char = result.get('recognized_char')
            
            # 提取原始序号
            import re
            match = re.search(r'char_(\d+)', filename)
            char_id = match.group(1) if match else f"{i:03d}"
            
            # 根据质量生成新文件名
            if quality == 'GOOD':
                if recognized_char:
                    # 清理文件名中不合法的字符
                    safe_char = self._sanitize_filename(recognized_char)
                    new_filename = f"Good_{safe_char}_{char_id}.png"
                else:
                    new_filename = f"Good_{char_id}.png"
                stats['good'] += 1
            elif quality == 'PARTIAL':
                new_filename = f"Partial_{char_id}.png"
                stats['partial'] += 1
            elif quality == 'MULTIPLE':
                new_filename = f"Multiple_{char_id}.png"
                stats['multiple'] += 1
            elif quality == 'NOISE':
                new_filename = f"Noise_{char_id}.png"
                stats['noise'] += 1
            elif quality == 'UNCLEAR':
                new_filename = f"Unclear_{char_id}.png"
                stats['unclear'] += 1
            else:  # ERROR或其他
                new_filename = f"Error_{char_id}.png"
                stats['error'] += 1
            
            # 复制文件到VL输出目录
            new_path = os.path.join(vl_output_dir, new_filename)
            try:
                import shutil
                shutil.copy2(old_path, new_path)
                stats['processed_files'].append({
                    'original_name': filename,
                    'vl_name': new_filename,
                    'quality': quality,
                    'original_path': old_path,
                    'vl_path': new_path
                })
                print(f"  ✅ {filename} -> {new_filename} ({quality})")
            except Exception as e:
                print(f"  ❌ 复制失败: {e}")
                stats['error'] += 1
            
            # 简单的速率控制
            if i % 5 == 0 and i < len(char_files):
                time.sleep(1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n=== VL评估与筛选完成 ===")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"处理速度: {len(char_files)/total_time:.2f}个/秒")
        print(f"\n质量统计:")
        print(f"  GOOD:     {stats['good']} ({stats['good']/len(char_files)*100:.1f}%)")
        print(f"  PARTIAL:  {stats['partial']} ({stats['partial']/len(char_files)*100:.1f}%)")
        print(f"  MULTIPLE: {stats['multiple']} ({stats['multiple']/len(char_files)*100:.1f}%)")
        print(f"  NOISE:    {stats['noise']} ({stats['noise']/len(char_files)*100:.1f}%)")
        print(f"  UNCLEAR:  {stats['unclear']} ({stats['unclear']/len(char_files)*100:.1f}%)")
        print(f"  ERROR:    {stats['error']} ({stats['error']/len(char_files)*100:.1f}%)")
        
        # 保存评估日志
        log_file = os.path.join(vl_output_dir, 'vl_evaluation_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_directory': char_dir,
                'output_directory': vl_output_dir,
                'statistics': stats,
                'model': self.config['model']
            }, f, ensure_ascii=False, indent=2)
        
        print(f"📝 评估日志已保存: {log_file}")
        print(f"📁 VL筛选结果保存在: {vl_output_dir}")
        
        return stats

    def evaluate_batch(self, char_image_paths: List[str], 
                      batch_size: int = 10,
                      save_results: bool = True) -> List[Dict[str, Any]]:
        """
        批量评估字符图片质量
        
        :param char_image_paths: 字符图片路径列表
        :param batch_size: 批次大小（暂时未用，逐个处理避免API限制）
        :param save_results: 是否保存结果
        :return: 评估结果列表
        """
        results = []
        total = len(char_image_paths)
        
        print(f"\n=== 开始批量字符质量评估 ===")
        print(f"待评估字符数量: {total}")
        print(f"模型: {self.config['model']}")
        
        start_time = time.time()
        
        for i, char_path in enumerate(char_image_paths, 1):
            print(f"\n[{i}/{total}] 评估: {os.path.basename(char_path)}")
            
            result = self.evaluate_character(char_path)
            results.append(result)
            
            # 显示进度
            quality = result['quality']
            is_good = result['is_good']
            status = "✅" if is_good else "❌"
            print(f"  结果: {quality} {status}")
            
            # 简单的速率控制，避免API限制
            if i % 5 == 0:
                time.sleep(1)  # 每5个字符休息1秒
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 统计结果
        stats = self._calculate_stats(results)
        
        print(f"\n=== 批量评估完成 ===")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"平均每个字符: {total_time/total:.2f}秒")
        print(f"\n质量统计:")
        for quality, count in stats['quality_counts'].items():
            percentage = (count / total) * 100
            print(f"  {quality}: {count} ({percentage:.1f}%)")
        
        print(f"\n好字符率: {stats['good_rate']:.1f}% ({stats['good_count']}/{total})")
        
        # 保存结果
        if save_results:
            self._save_batch_results(results, stats)
        
        return results
    
    def _calculate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算评估统计信息
        """
        from collections import Counter
        
        quality_counts = Counter([r['quality'] for r in results])
        good_count = sum(1 for r in results if r['is_good'])
        total_count = len(results)
        good_rate = (good_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'total_count': total_count,
            'good_count': good_count,
            'good_rate': good_rate,
            'quality_counts': dict(quality_counts)
        }
    
    def _save_batch_results(self, results: List[Dict[str, Any]], 
                           stats: Dict[str, Any]) -> str:
        """
        保存批量评估结果
        """
        ensure_dir(VL_EVALUATIONS_DIR)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(VL_EVALUATIONS_DIR, f"character_evaluation_{timestamp}.json")
        
        output_data = {
            'evaluation_info': {
                'timestamp': timestamp,
                'model': self.config['model'],
                'total_characters': len(results)
            },
            'statistics': stats,
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"📝 评估结果已保存: {output_path}")
        return output_path


def filter_segment_characters(segment_dir: str, vl_output_dir: str = None) -> Dict[str, Any]:
    """
    便捷函数：对segment目录中的字符进行VL评估和筛选到VL目录
    
    :param segment_dir: segment结果目录路径
    :param vl_output_dir: VL输出目录，如果为None则自动生成
    :return: 评估统计信息
    """
    evaluator = VLCharacterEvaluator()
    return evaluator.evaluate_and_filter_characters(segment_dir, vl_output_dir)


def evaluate_character_directory(char_dir: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    便捷函数：评估目录中的所有字符图片（不重命名）
    
    :param char_dir: 字符图片目录
    :param limit: 限制评估数量（用于测试）
    :return: 评估结果列表
    """
    if not os.path.exists(char_dir):
        print(f"❌ 字符目录不存在: {char_dir}")
        return []
    
    # 获取所有字符图片
    char_paths = []
    for file in sorted(os.listdir(char_dir)):
        if file.startswith('char_') and file.endswith('.png'):
            char_paths.append(os.path.join(char_dir, file))
    
    if limit:
        char_paths = char_paths[:limit]
    
    print(f"找到 {len(char_paths)} 个字符图片")
    
    if not char_paths:
        print("❌ 没有找到字符图片")
        return []
    
    # 创建评估器并评估
    evaluator = VLCharacterEvaluator()
    results = evaluator.evaluate_batch(char_paths)
    
    return results


if __name__ == '__main__':
    # 测试用例：对segment结果进行VL评估和筛选
    test_segment_dir = "/Users/yinwhe/Desktop/Fbb/DataVis/charsis/data/results/segments/demo_preprocessed_alpha1.5_beta10"
    
    print("=== VL字符评估与筛选测试 ===")
    print(f"segment目录: {test_segment_dir}")
    
    # 对segment目录中的字符进行VL评估和筛选到VL目录
    stats = filter_segment_characters(test_segment_dir)
    
    if 'error' not in stats:
        print(f"\n🎉 筛选完成！")
        print(f"处理文件数: {stats['total']}")
        print(f"Good文件: {stats['good']}")
    else:
        print(f"❌ 处理失败: {stats['error']}")