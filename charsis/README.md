# Charsis - 刻本图片文字处理与分析

本项目旨在对刻本图片进行一系列处理与分析，包括图像预处理、文字切割、OCR识别以及数据分析。

## 项目结构

```
/Users/yinwhe/Desktop/Fbb/DataVis/charsis/
├── data/
│   ├── raw/                # 存放原始输入图片
│   └── results/            # 统一的结果输出目录
│       ├── preprocessed/   # 预处理结果
│       ├── segments/       # 文字切割结果
│       ├── ocr/            # OCR识别结果
│       └── analysis/       # 分析结果
├── notebooks/              # 用于探索性数据分析的 Jupyter Notebooks
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   ├── preprocess/         # 图像预处理模块
│   ├── segmentation/       # 文字切割模块
│   ├── ocr/                # 文字识别模块
│   ├── analysis/           # 数据分析模块
│   └── utils/             # 通用工具
├── tests/                  # 存放单元测试
├── requirements.txt
└── README.md               # 项目说明文档
```

## 使用说明

(待补充)