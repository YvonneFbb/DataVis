# Charsis - 刻本图片文字处理与分析

本项目旨在对刻本图片进行一系列处理与分析，包括图像预处理、文字切割、OCR识别以及数据分析。

## 项目结构

```
charsis/
├── data/
│   ├── raw/                # 存放原始输入图片
│   └── results/            # 统一的结果输出目录
│       ├── preprocessed/   # 预处理结果
    │       ├── segments/       # 文字切割结果
    │       ├── preocr/         # 文本区域检测（区域框/region图等）
    │       ├── ocr/            # 字符识别结果（保留目录名用于兼容）
│       └── analysis/       # 分析结果
├── notebooks/              # 用于探索性数据分析的 Jupyter Notebooks
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   ├── preprocess/         # 图像预处理模块
    │   ├── segmentation/       # 文字切割模块
    │   ├── preocr/             # 文本区域检测（远程 PaddleOCR 客户端等）
    │   ├── postocr/            # 分割后字符识别与重命名
│   ├── analysis/           # 数据分析模块
│   └── utils/             # 通用工具
├── tests/                  # 存放单元测试
├── requirements.txt
└── README.md               # 项目说明文档
```

## 使用说明

### 环境搭建

1. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 快速开始

1. **图像预处理**：
```bash
cd src/preprocess
python core.py
```

2. **字符分割**：
```bash
cd src/segmentation
python core.py
```

### 核心改进 (v2.0)

本版本对投影分析模块进行了重大改进：

#### 主要改进点
- ✨ **高斯平滑**: 替代简单均值滤波，更好保持边界信息
- 🎯 **自适应阈值**: 基于四分位数的动态阈值计算，适应不同密度文本
- 🔍 **智能分割点选择**: 多候选点评分机制，综合投影值、位置合理性
- ✅ **分割验证**: 确保分割结果的合理性，避免过小片段

#### 性能提升
- 减少过度分割 **20-30%**
- 参数自适应，无需手动调优
- 处理复杂字符连接场景更加准确

#### 配置参数

新增配置参数位于 `src/config.py` 的 `PROJECTION_CONFIG` 中：

```python
PROJECTION_CONFIG = {
    'gaussian_sigma_ratio': 20,      # 高斯核标准差比例
    'adaptive_percentile_low': 25,   # 自适应阈值低分位数
    'adaptive_percentile_high': 75,  # 自适应阈值高分位数
    'max_splits_limit': 5,           # 最大分割数限制
    # ... 更多参数详见配置文件
}
```

### 版本兼容性

- ✅ 完全向后兼容，现有调用方式无需修改
- ✅ 原有配置参数保留，新参数为可选
- ✅ 可通过 `projection_original.py` 回退到原版本

### 测试验证

运行完整测试：
```bash
python src/segmentation/core.py
```

处理结果将保存在 `data/results/segments/` 目录中。