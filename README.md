# YOLO-MHAF: Multi-Head Adaptive Fusion for Peanut Leaf Disease Detection

This repository implements an improved YOLOv8n-based detection model named YOLO-MHAF featuring:

- Mixed lightweight backbone with Ghost modules in shallow stages and SPD-Conv in deeper stages
- C2f-Hybrid block (Coordinate Attention + ECA) replacing the C2f above SPPF
- AMFN neck combining BiFPN and CARAFE for adaptive multi-path feature fusion

The code provides the full model definition in PyTorch and a smoke test. You can integrate the model into your own training pipeline or extend the provided training skeleton.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo (forward only)

```bash
python scripts/demo_forward.py
```

This will construct the model and run a random 640x640 image through it, printing output shapes of the three detection scales.

## Project structure

- `yolomhaf/` core model modules
  - `attention.py`: Coordinate Attention (CA) and Efficient Channel Attention (ECA)
  - `blocks.py`: Conv/BNSiLU (CBS), GhostConv, SPD-Conv, C2f and C2fHybrid, SPPF
  - `carafe.py`: CARAFE upsampling operator (PyTorch implementation)
  - `bifpn.py`: BiFPN fusion block with learnable weights
  - `amfn.py`: AMFN neck using CARAFE + BiFPN
  - `backbone.py`: YOLOv8n-style backbone with Ghost+SPD-Conv and C2fHybrid above SPPF
  - `head.py`: Lightweight decoupled detection head (anchor-free output tensor)
  - `yolomhaf.py`: Full YOLO-MHAF model wiring backbone, neck and head
- `scripts/demo_forward.py`: simple forward pass demo
- `tests/smoke_test.py`: quick unit test for tensor shapes

## Notes

- The model returns three prediction tensors (P3, P4, P5) as `[B, C, H, W]`, where `C = num_classes + 5` (bbox 4 + objectness 1 + classes). You can adapt the head to match your preferred training targets.
- Extend `train.py` as needed to integrate with your dataset and losses.
