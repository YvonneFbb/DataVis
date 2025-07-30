import os

def ensure_dir(path):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path

def get_output_path(input_path, output_dir, suffix=None):
    """根据输入路径生成对应的输出路径
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出根目录
        suffix: 输出文件后缀，默认使用原文件后缀
    """
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    if suffix:
        ext = suffix if suffix.startswith('.') else f'.{suffix}'
    return os.path.join(ensure_dir(output_dir), f"{name}{ext}")