# 轮廓提取脚本使用说明

该脚本读取输入图像，输出：
- 二值掩膜：`process/out/<name>_mask.png`
- 轮廓叠加：`process/out/<name>_overlay.png`
- SVG 多边形：`process/out/<name>_contours.svg`
 - 纹理边缘：`process/out/<name>_texture_edges.png`
 - 纹理叠加：`process/out/<name>_texture_overlay.png`

## 运行方式

方式一：使用配置文件（推荐，带注释可调；省略 --config 时会自动查找 process/contours_config.toml）

```bash
/Users/yinwhe/Desktop/Fbb/DataVis/flower/.venv/bin/python process/extract_contours.py --config process/contours_config.toml
```

方式二：命令行参数（会覆盖配置文件中的值）

```bash
/Users/yinwhe/Desktop/Fbb/DataVis/flower/.venv/bin/python process/extract_contours.py \
  --input process/demo.png \
  --output-dir process/out \
  --method auto \
  --invert auto \
  --keep-biggest \
  --external \
  --save-svg \
  --chain none \
  --simplify 0
```

## 参数与配置

你可以在 `process/contours_config.toml` 中编辑同名字段，或用命令行覆盖：

- `--input`、`--output-dir`
- `--method`: auto | otsu | adaptive | canny
- `--invert`: auto | true | false
- `--min-area`: 过滤小轮廓（占图像面积比例），默认 0.005
- `--bilateral`: 双边滤波（默认开启，保边去噪）
- `--morph`、`--close-k --open-k --close-iter --open-iter`: 形态学清理（默认开启，核 7/5）
- `--external`: 仅外轮廓
- `--keep-biggest`: 仅保留最大连通域
- `--chain`: none | simple | tc89 | tc89_l1（none 为逐像素，最贴合掩膜）
- `--simplify`: RDP 简化强度（0 关闭，最贴合；>0 更平滑）
- `--thickness`: 叠加线条粗细
- `--save-svg`: 导出 SVG 多边形
 
### 图像增强（对齐你给的流程示例）
- `--enhance`：开启图像增强，包含：
  - CLAHE（亮度对比度增强，在 YCrCb 的 Y 通道）
  - HSV 饱和度提升（`--sat-factor`）
  - Gamma 校正（`--gamma`）
- `--use-enhanced-for-mask`：掩膜分割是否基于增强后的图

### 纹理边缘提取
- `--texture`：开启纹理边缘输出
- `--texture-method`：canny | sobel | laplacian | fusion
- `--use-enhanced-for-texture`：纹理是否基于增强后的图
- 其他：`--texture-sigma`、`--texture-sobel-ksize`、`--texture-lap-ksize`、`--texture-thresh`、`--texture-close-k`、`--texture-blur-ksize`

## 小贴士

- 背景不均匀：`--method adaptive`
- 边缘细易断：`--method canny` 并保留 `--morph`
- 噪声过多：增大 `--min-area`（如 0.01）
- 轮廓要更贴合掩膜：`--chain none --simplify 0`
- 轮廓更顺滑但略有近似：调高 `--simplify`（如 0.02–0.05）或增大 `--close-k / --open-k`
