# Charsis - 刻本图片文字处理与分析

本项目提供刻本图片的端到端处理流水线：预处理 → 文本区域检测（preOCR）→ 字符分割 → 字符识别（postOCR），并附带可选的分析与可视化。

## 目录结构

```
charsis/
├── data/
│   ├── raw/                         # 原始输入图片
│   └── results/
│       ├── preprocessed/            # 预处理结果（单图或按基名分文件夹）
│       ├── preocr/                  # 文本区域检测（regions.json、region_images/）
│       ├── segments/                # 字符分割结果（每个 region 一个文件夹）
│       ├── ocr/                     # 分割后的字符识别与重命名结果
│       └── analysis/                # 分析结果
├── src/
│   ├── config.py                    # 全局配置（路径、预处理/分割参数、OCR配置）
│   ├── preprocess/core.py           # 预处理 CLI
│   ├── preocr/remote_paddleocr.py   # 远程 PaddleOCR 客户端
│   ├── segmentation/vertical_hybrid.py  # 垂直单列字符分割（支持批处理）
│   ├── postocr/core.py              # 分割后字符识别与重命名
│   ├── analysis/                    # 可选分析
│   └── utils/                       # 工具
├── hybrid_pipeline.py               # 一键式流水线（可跳过任意阶段）
├── requirements.txt
└── README.md
```

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

可选环境变量：
- PPOCR_SERVER_URL：远程 PaddleOCR 服务地址（默认读取 `src/config.py` 中 OCR_REMOTE_CONFIG，或环境变量覆盖）
- PPOCR_TIMEOUT：请求超时秒数

## 分阶段使用说明

下面分别介绍每个阶段的最小命令与常用参数。

### 1) 预处理 preprocess

- 基线：
```bash
python src/preprocess/core.py \
    --image data/raw/demo.jpg \
    --output-dir data/results/preprocessed
```

- 启用“断笔/小缺口修补”并输出精简调试图（3 张：heal 结果、diff 热力、增补叠加）：
```bash
python src/preprocess/core.py \
    --image data/raw/demo.jpg \
    --output-dir data/results/preprocessed \
    --heal --heal-debug
```

- 修补强度更明显（如默认不明显时）：
```bash
python src/preprocess/core.py \
    --image data/raw/demo.jpg \
    --output-dir data/results/preprocessed \
    --heal --heal-debug \
    --heal-kernel 5 --heal-iterations 2 --heal-directions iso,h,v,d1,d2
```

- 批量处理整个目录：
```bash
python src/preprocess/core.py \
    --input-dir data/raw \
    --output-dir data/results/preprocessed \
    --heal --heal-debug
```

相关配置（`src/config.py`）：
- PREPROCESS_STROKE_HEAL_CONFIG：断笔修补（方向核 + 可选双边滤波）
- PREPROCESS_INK_PRESERVE_CONFIG：墨色保持/增强（黑帽回墨 + 可选反锐化，避免整体发灰）

输出：`data/results/preprocessed/<基名>/enhanced.jpg`（在流水线模式），或你指定的输出文件。

### 2) 文本区域检测 preOCR（基于远程 PaddleOCR）

推荐通过流水线运行（见下文“一键流水线”）。如仅运行到 preOCR：
```bash
python hybrid_pipeline.py --input data/raw/demo.jpg --skip-segment
```
输出到 `data/results/preocr/<基名>/`：
- regions.json：区域元数据（rect_bbox、confidence、text 等）
- region_images/region_XXX.jpg：每个检测到的文本区域截图

### 3) 字符分割 segmentation（垂直单列）

- 单张 region 图：
```bash
python src/segmentation/vertical_hybrid.py \
    --image data/results/preocr/demo/region_images/region_001.jpg \
    --out   data/results/segments/demo/region_001
```
可选参数：
- `--no-seam` 关闭动态 seam 细化
- `--no-align` 关闭基于期望字数的 split/merge 对齐（通常保持开启）

- 批量处理 preOCR 的所有 region 图：
```bash
python src/segmentation/vertical_hybrid.py --scan-ocr --dataset demo
```
输出：每个 region 写到 `data/results/segments/<dataset>/<region_xxx>/`，包含：
- `char_*.png`：裁剪后的字符切片（已做紧致裁边与少量噪声清理）
- `overlay.png`：分割框与 seam 叠加图
- `summary.json`：该 region 的分割统计

相关配置（`src/config.py`）：
- SEGMENT_REFINEMENT_CONFIG：seam 带宽/代价权重、期望数量对齐
- CHAR_CROP_CONFIG：切片内容裁边/留白/是否方形
- CHAR_NOISE_CLEAN_CONFIG：连通域噪点清理、形态学微调

### 4) 字符识别与重命名 postOCR

- 处理单个切片目录：
```bash
python src/postocr/core.py \
    --input-dir data/results/segments/demo/region_001 \
    --output-dir data/results/ocr/demo_region_001
```

- 批量处理 `data/results/segments/` 下所有含 `char_*.png` 的目录：
```bash
python src/postocr/core.py
```
输出：对应输出目录内的重命名图片与 `ocr_results.json`；支持 `--conf-thres` 调整低置信度阈值。

## 一键式流水线（推荐）

```bash
python hybrid_pipeline.py --input data/raw/demo.jpg
```
可跳过任意阶段：
```bash
python hybrid_pipeline.py --input data/raw/demo.jpg \
    --skip-preprocess   # 直接用原图做 preOCR
    --skip-ocr          # 使用已有 preOCR 结果做分割
    --skip-segment      # 仅做预处理+preOCR
```

流水线输出：
- `data/results/preprocessed/<基名>/enhanced.jpg`
- `data/results/preocr/<基名>/{regions.json, annotated.jpg, region_images/}`
- `data/results/segments/<基名>/<region_xxx>/`
- `data/results/segments/<基名>/all_characters/`（所有字符汇总拷贝）
- `data/results/segments/<基名>/pipeline_summary.json`

## 调参与常见问题

- 预处理后“背景干净但笔画变淡”：
    - 在 `src/config.py` 调整 `PREPROCESS_INK_PRESERVE_CONFIG`：
        - 降低 `blackhat_strength`（如 0.4~0.5）或调小 `blackhat_kernel`；如仍偏灰，略增至 0.7 或核=11。
        - 关闭或减小 `unsharp_amount` 防止过锐导致噪点。

- 修补（heal）不明显 / 过度粘连：
    - 不明显：`--heal-kernel 5 --heal-iterations 2 --heal-directions iso,h,v,d1,d2`
    - 粘连：回退为 `--heal-kernel 3 --heal-iterations 1 --heal-directions iso,h,v`

- 分割数量与期望不符：
    - 流水线会用 preOCR 的 `text` 作为期望字数对齐；也可在单图模式传 `--expected-text` 给分割函数（内部使用）。

- 远程 OCR 地址：
    - 设置环境变量 `PPOCR_SERVER_URL`，或修改 `src/config.py` 中 `OCR_REMOTE_CONFIG`。

## 版本与说明

- 当前默认分割器为 `src/segmentation/vertical_hybrid.py`，支持动态 seam 与期望数量对齐。
- 历史的投影参数（老版 PROJECTION_CONFIG）已移除；若需回退旧算法，请检索历史版本记录。

---

有任何问题或改进想法，欢迎提 Issue 或直接标注希望优化的样例区域，我们会针对性调参和迭代算法。