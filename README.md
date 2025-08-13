# GLTR-SDNet: Infrared Small Target Detection (CNN + Transformer)

GLTR-SDNet is a global-local collaborative model for single-frame infrared small target detection/segmentation. It integrates:

- Detail-Preserving High-Pass Convolution (DPHC) for local contrast enhancement
- Saliency-guided Dynamic Tokenization (SDT) to prune background tokens
- Global-Local Cross Attention (GLCA) combining window self-attention and global memory cross-attention
- Semantic Gated Fusion (SGF) for top-down decoder fusion
- Dynamic Query Refinement (DQR) for small-region refinement

The codebase layout loosely follows YOLOv5-style structure.

## Quickstart

### 1) Environment

Recommended Python: 3.10–3.12 (for stable PyTorch wheels)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If your system Python is 3.13 and PyTorch wheels are unavailable, consider creating a 3.12 environment (e.g., via conda or pyenv) before installing.

### 2) Train on synthetic data (sanity-check)

```bash
python -m gltr_sdnet.train --config gltr_sdnet/configs/default.yaml --dataset synthetic --epochs 1 --img-size 256 --batch-size 4
```

### 3) Train on a real dataset (IRSTD-1k or NUDT-SIRST)

Expected folder layout:

```
DATA_ROOT/
  images/
    xxx.png (or .jpg)
  masks/
    xxx.png (binary mask 0/255)
```

Run:

```bash
python -m gltr_sdnet.train --config gltr_sdnet/configs/default.yaml \
       --dataset custom --data-root /path/to/DATA_ROOT \
       --epochs 120 --img-size 512 --batch-size 8
```

### 4) Validation

```bash
python -m gltr_sdnet.val --weights runs/exp/weights/best.pt --data-root /path/to/DATA_ROOT
```

### 5) Inference

```bash
python -m gltr_sdnet.infer --weights runs/exp/weights/best.pt --source /path/to/image_or_dir --save-dir runs/infer
```

## Configuration

See `gltr_sdnet/configs/default.yaml`. Key options include:
- model: channels, window size, token ratio, number of memory tokens
- train: optimizer, lr, weight decay, epochs, augmentations
- loss: focal, iou, boundary, background weights

## Notes
- The repository includes a synthetic dataset generator for smoke tests.
- The implementation targets clarity and readability first; some modules are simplified but faithful to the described design.
- For deployment on edge devices, consider exporting to TorchScript or ONNX and applying quantization/pruning.
