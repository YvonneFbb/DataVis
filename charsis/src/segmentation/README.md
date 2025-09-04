# Segmentation (Recreated)

This module provides a robust vertical single-column segmentation tuned for preOCR region images (e.g., `data/results/preocr/demo/region_images/region_001.jpg`).

- `vertical_hybrid.py`: main implementation and a CLI for quick testing.

Usage:
- Run directly to test on region_001:

```
python src/segmentation/vertical_hybrid.py
```

- Or import and call `run_on_image(path, out_dir, expected_text=None)`
