# 纹理边缘提取（精简版）

此工具仅做“图片优化 + 纹理边缘”两步，输出：
- `<name>_texture_edges.png`
- `<name>_texture_overlay.png`

## 快速运行（省略 --config 时会自动查找 process/texture_config.toml）
```bash
/Users/yinwhe/Desktop/Fbb/DataVis/flower/.venv/bin/python process/extract_texture.py --config process/texture_config.toml
```

## 可调参数（也可在 `texture_config.toml` 设置）
- 图像增强：`enhance`、`clahe_clip`、`clahe_tile`、`sat_factor`、`gamma`、`use_enhanced`
- 纹理方法：
  - `texture_method`: canny | sobel | laplacian | fusion
  - `texture_sigma`（canny）
  - `texture_sobel_ksize`、`texture_lap_ksize`
  - `texture_thresh`（sobel/laplacian 阈值，或 "auto"=Otsu）
  - `texture_close_k`（闭运算连通边缘）
  - `texture_blur_ksize`（预模糊抑制噪声）

## 调参建议
- 纹理更丰富：`texture_method=fusion`，`texture_close_k=5`
- 纹理更干净：提高 `texture_thresh`（如 80）、减小 `sat_factor`、增大 `texture_blur_ksize`
- 细节更清晰：适度提高 `clahe_clip` 或 `gamma`（>1.0 会提亮中间调）
