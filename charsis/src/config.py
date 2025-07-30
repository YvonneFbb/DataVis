import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# 结果目录
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, 'preprocessed')
SEGMENTS_DIR = os.path.join(RESULTS_DIR, 'segments')
OCR_DIR = os.path.join(RESULTS_DIR, 'ocr')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')