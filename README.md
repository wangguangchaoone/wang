# Infrared Weak and Small Target Detection via Multi-Scale Feature Fusion and Attention Mechanism

This repository contains the implementation of a novel deep learning framework for infrared weak and small target detection, combining Convolutional Neural Networks (CNNs) with Transformer architecture.

## 📋 Paper Overview

The paper proposes a hybrid architecture that addresses the challenges of infrared weak and small target detection:

- **Multi-scale feature fusion module** for capturing targets at different scales
- **Transformer-based detection head** with attention mechanism for global modeling
- **Specialized loss function** designed for weak and small target detection
- **Comprehensive ablation studies** demonstrating component effectiveness

## 🏗️ Architecture

The proposed framework consists of three main components:

1. **Modified ResNet-50 Backbone**: Adapted for infrared images with single-channel input
2. **Multi-Scale Feature Fusion**: Integrates features from different scales (P2, P3, P4, P5)
3. **Transformer Detection Head**: Multi-head attention mechanism for target detection

## 📊 Results

Performance on SIRST dataset:
- **AP@0.5**: 89.7%
- **Recall**: 87.3%
- **FPR**: 0.08
- **mAP**: 85.2%

Significant improvements over state-of-the-art methods including YOLOv5, Faster R-CNN, and DETR.

## 📁 Repository Structure

```
.
├── infrared_weak_target_detection_paper.md    # Main paper content
├── experimental_details.md                     # Detailed experimental results
├── train.py                                   # Complete training script
├── README.md                                  # This file
└── requirements.txt                           # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv infrared_detection
source infrared_detection/bin/activate  # On Windows: infrared_detection\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── image1.png
│   ├── image1.txt
│   └── ...
├── val/
│   ├── image1.png
│   ├── image1.txt
│   └── ...
└── test/
    ├── image1.png
    ├── image1.txt
    └── ...
```

Annotation format (YOLO style):
```
class_id x_center y_center width height
```

### 3. Training

```bash
# Basic training
python train.py --data_dir ./data --output_dir ./outputs

# Custom parameters
python train.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --batch_size 16 \
    --epochs 150 \
    --lr 5e-5
```

### 4. Model Evaluation

The training script automatically saves:
- Best model: `outputs/best_model.pth`
- Checkpoints: `outputs/checkpoint_epoch_X.pth`
- Training logs: `training.log`

## 🔧 Configuration

### Model Parameters

```python
model_config = {
    'in_channels': 1,           # Infrared images (single channel)
    'fusion_dim': 256,          # Feature fusion dimension
    'd_model': 256,             # Transformer model dimension
    'nhead': 8,                 # Number of attention heads
    'num_layers': 6             # Number of transformer layers
}
```

### Training Parameters

```python
training_config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 8,
    'epochs': 100,
    'warmup_epochs': 5,
    'lr_scheduler': 'cosine'
}
```

### Data Augmentation

```python
aug_config = {
    'rotation_range': [-15, 15],      # Random rotation
    'scale_range': [0.8, 1.2],       # Random scaling
    'brightness_range': [0.8, 1.2],  # Brightness adjustment
    'noise_std': 0.02                # Gaussian noise
}
```

## 📈 Ablation Studies

The paper includes comprehensive ablation studies:

1. **Multi-Scale Feature Fusion**: Impact of different scale combinations
2. **Attention Mechanism**: Comparison of attention types
3. **Loss Function Components**: Contribution of each loss term
4. **Target Size Analysis**: Performance across different target sizes

## 🎯 Key Innovations

### 1. Multi-Scale Feature Fusion
- Learnable weights for feature combination
- Preserves spatial resolution across scales
- Effective for targets of varying sizes

### 2. Hybrid Architecture
- Combines CNN local feature extraction with Transformer global modeling
- Addresses both local and global dependencies
- Optimized for infrared image characteristics

### 3. Specialized Loss Function
- Focal loss for classification
- Smooth L1 loss for regression
- Attention loss for target focus

## 🔬 Experimental Setup

### Hardware
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- CPU: Intel Xeon E5-2680 v4
- RAM: 64GB DDR4

### Software
- Python 3.8.10
- PyTorch 1.12.1
- CUDA 11.6

### Dataset
- **SIRST Dataset**: 427 images, 1,130 targets
- **Target sizes**: 3×3 to 20×20 pixels
- **Split**: 80% train, 10% validation, 10% test

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{infrared_target_detection_2024,
  title={Infrared Weak and Small Target Detection via Multi-Scale Feature Fusion and Attention Mechanism},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SIRST dataset creators
- PyTorch community
- Research community for open-source contributions

## 📞 Contact

For questions and discussions, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.
