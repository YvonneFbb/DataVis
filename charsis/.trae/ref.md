# 项目参考

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

## 开发计划

1.  **图像预处理 (`src/preprocess`)**: 实现图像的灰度化、二值化、去噪等功能。
2.  **文字切割 (`src/segmentation`)**: 将预处理后的图片中的文字进行单字切割。
3.  **OCR识别 (`src/ocr`)**: 对切割后的单字图片进行文字识别。
4.  **数据分析 (`src/analysis`)**: 对识别结果进行统计和分析。