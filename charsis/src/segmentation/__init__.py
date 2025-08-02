"""
字符分割模块
提供字符检测、分割、合并和可视化功能
"""

from .core import segment_characters, process_all_preprocessed_images
from .detection import (
    analyze_character_dimensions,
    detect_vertically_connected_chars,
    detect_wide_chars,
    reclassify_split_character
)
from .projection import (
    horizontal_projection_split_vertical_chars,
    vertical_projection_split_wide_chars
)
from .merger import (
    merge_narrow_characters,
    calculate_robust_dimensions,
    classify_characters,
    red_red_merge,
    red_green_merge
)
from .visualization import (
    annotate_characters,
    save_annotated_image,
    print_classification_stats,
    create_legend_image
)

__all__ = [
    # 主要接口
    'segment_characters',
    'process_all_preprocessed_images',
    
    # 字符检测
    'analyze_character_dimensions',
    'detect_vertically_connected_chars',
    'detect_wide_chars',
    'reclassify_split_character',
    
    # 投影分析
    'horizontal_projection_split_vertical_chars',
    'vertical_projection_split_wide_chars',
    
    # 字符合并
    'merge_narrow_characters',
    'calculate_robust_dimensions',
    'classify_characters',
    'red_red_merge',
    'red_green_merge',
    
    # 可视化
    'annotate_characters',
    'save_annotated_image',
    'print_classification_stats',
    'create_legend_image'
]