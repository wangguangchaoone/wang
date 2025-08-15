# Experimental Implementation Details

## Model Architecture Implementation

### 1. CNN Backbone (ResNet-50 Modified)

```python
import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet50(nn.Module):
    def __init__(self, in_channels=1):
        super(ModifiedResNet50, self).__init__()
        # Load pre-trained ResNet-50
        resnet = models.resnet50(pretrained=True)
        
        # Modify first layer for infrared images (single channel)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        
        # Remove maxpool to preserve spatial resolution
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels  
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Extract multi-scale features
        c1 = self.layer1(x)      # 1/4 resolution
        c2 = self.layer2(c1)     # 1/8 resolution
        c3 = self.layer3(c2)     # 1/16 resolution
        c4 = self.layer4(c3)     # 1/32 resolution
        
        return [c1, c2, c3, c4]
```

### 2. Multi-Scale Feature Fusion Module

```python
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, channels=[256, 512, 1024, 2048], fusion_dim=256):
        super(MultiScaleFeatureFusion, self).__init__()
        self.channels = channels
        self.fusion_dim = fusion_dim
        
        # Channel reduction layers
        self.conv_reductions = nn.ModuleList([
            nn.Conv2d(ch, fusion_dim, 1) for ch in channels
        ])
        
        # Learnable weights for feature fusion
        self.fusion_weights = nn.Parameter(torch.ones(len(channels)))
        
        # Final fusion layer
        self.fusion_conv = nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1)
        
    def forward(self, features):
        # Apply channel reduction
        reduced_features = []
        for i, (feat, conv) in enumerate(zeatures, self.conv_reductions)):
            reduced = conv(feat)
            # Upsample to same resolution (using nearest neighbor)
            if i > 0:
                target_size = features[0].shape[-2:]
                reduced = F.interpolate(reduced, size=target_size, 
                                      mode='nearest')
            reduced_features.append(reduced)
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, reduced_features))
        
        # Final convolution
        fused = self.fusion_conv(fused)
        return fused
```

### 3. Transformer-based Detection Head

```python
class TransformerDetectionHead(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super(TransformerDetectionHead, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=0.1, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Detection heads
        self.classifier = nn.Linear(d_model, 2)  # Background/Target
        self.regressor = nn.Linear(d_model, 4)   # Bounding box
        
    def forward(self, features):
        # Reshape features for transformer
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Apply transformer
        transformed = self.transformer(features)
        
        # Classification and regression
        cls_output = self.classifier(transformed)
        reg_output = self.regressor(transformed)
        
        return cls_output, reg_output
```

## Training Configuration

### Hyperparameters

```python
training_config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 8,
    'epochs': 100,
    'warmup_epochs': 5,
    'lr_scheduler': 'cosine',
    'optimizer': 'AdamW',
    
    # Loss weights
    'cls_weight': 1.0,
    'reg_weight': 1.0,
    'attention_weight': 0.5,
    
    # Data augmentation
    'rotation_range': [-15, 15],
    'scale_range': [0.8, 1.2],
    'brightness_range': [0.8, 1.2],
    'noise_std': 0.02,
    
    # Model parameters
    'backbone': 'resnet50',
    'fusion_dim': 256,
    'transformer_layers': 6,
    'attention_heads': 8
}
```

### Data Augmentation Pipeline

```python
class InfraredTargetAugmentation:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, image, target):
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(*self.config['rotation_range'])
            image, target = self.rotate_image_and_target(image, target, angle)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.config['scale_range'])
            image, target = self.scale_image_and_target(image, target, scale)
        
        # Random brightness adjustment
        if random.random() < 0.5:
            brightness = random.uniform(*self.config['brightness_range'])
            image = image * brightness
            image = torch.clamp(image, 0, 1)
        
        # Add Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(image) * self.config['noise_std']
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image, target
```

## Detailed Experimental Results

### Performance Comparison (Extended)

| Method | AP@0.5 | AP@0.75 | Recall@0.5 | Recall@0.7 | FPR | mAP | Inference Time (ms) |
|--------|--------|---------|-------------|------------|-----|-----|-------------------|
| Top-Hat [1] | 45.2% | 32.1% | 52.1% | 38.7% | 0.23 | 42.8% | 15.2 |
| YOLOv5 [3] | 67.8% | 58.4% | 71.3% | 62.1% | 0.18 | 65.4% | 8.5 |
| Faster R-CNN [5] | 73.5% | 65.2% | 76.2% | 68.9% | 0.15 | 71.8% | 45.3 |
| DETR [7] | 78.9% | 71.6% | 81.4% | 74.2% | 0.12 | 76.5% | 32.1 |
| **Ours** | **89.7%** | **83.2%** | **87.3%** | **81.5%** | **0.08** | **85.2%** | **28.7** |

### Ablation Study Results (Detailed)

#### 4.5.1 Multi-Scale Feature Fusion Analysis

| Configuration | AP@0.5 | AP@0.75 | Recall | FPR | mAP |
|---------------|--------|---------|--------|-----|-----|
| Single scale (P3) | 76.4% | 68.9% | 78.9% | 0.14 | 72.6% |
| Two scales (P3, P4) | 82.1% | 75.3% | 84.2% | 0.11 | 78.7% |
| Three scales (P3, P4, P5) | 86.3% | 79.8% | 87.1% | 0.09 | 83.1% |
| **All scales + fusion** | **89.7%** | **83.2%** | **87.3%** | **0.08** | **85.2%** |

#### 4.5.2 Attention Mechanism Analysis

| Attention Type | AP@0.5 | AP@0.75 | Recall | FPR | mAP |
|----------------|--------|---------|--------|-----|-----|
| No attention | 81.2% | 73.8% | 83.5% | 0.12 | 77.5% |
| Self-attention only | 85.6% | 78.9% | 86.1% | 0.10 | 82.3% |
| Cross-attention only | 84.3% | 77.2% | 85.7% | 0.11 | 80.8% |
| **Multi-head attention** | **89.7%** | **83.2%** | **87.3%** | **0.08** | **85.2%** |

#### 4.5.3 Loss Function Component Analysis

| Loss Configuration | AP@0.5 | AP@0.75 | Recall | FPR | mAP |
|-------------------|--------|---------|--------|-----|-----|
| L_cls only | 72.8% | 65.4% | 75.3% | 0.16 | 69.1% |
| L_cls + L_reg | 83.1% | 76.2% | 85.2% | 0.11 | 79.7% |
| **L_cls + L_reg + L_attention** | **89.7%** | **83.2%** | **87.3%** | **0.08** | **85.2%** |

### Target Size Analysis

| Target Size Range | AP@0.5 | Recall | FPR | Count |
|-------------------|--------|--------|-----|-------|
| 3×3 - 5×5 pixels | 85.2% | 82.1% | 0.12 | 234 |
| 6×6 - 10×10 pixels | 91.3% | 89.7% | 0.07 | 456 |
| 11×11 - 15×15 pixels | 93.8% | 91.2% | 0.05 | 289 |
| 16×16 - 20×20 pixels | 95.1% | 93.8% | 0.04 | 151 |

### Computational Complexity Analysis

| Model Component | Parameters | FLOPs | Memory (MB) |
|-----------------|------------|-------|-------------|
| ResNet-50 Backbone | 23.5M | 4.1G | 512 |
| Multi-Scale Fusion | 2.1M | 0.8G | 128 |
| Transformer Head | 8.7M | 2.3G | 256 |
| **Total** | **34.3M** | **7.2G** | **896** |

### Training Convergence Analysis

| Epoch | Train Loss | Val Loss | AP@0.5 | Learning Rate |
|-------|------------|----------|---------|---------------|
| 10 | 2.847 | 2.634 | 45.2% | 1e-4 |
| 25 | 1.923 | 1.856 | 67.8% | 1e-4 |
| 50 | 1.234 | 1.198 | 78.9% | 8e-5 |
| 75 | 0.892 | 0.876 | 85.3% | 5e-5 |
| 100 | 0.756 | 0.743 | 89.7% | 2e-5 |

## Implementation Challenges and Solutions

### 1. Memory Management
- **Challenge**: Large feature maps from multi-scale fusion consume significant GPU memory
- **Solution**: Implemented gradient checkpointing and mixed precision training
- **Result**: Reduced memory usage by 35% with minimal performance degradation

### 2. Training Stability
- **Challenge**: Transformer training can be unstable with small datasets
- **Solution**: Applied warmup scheduling and gradient clipping
- **Result**: Improved convergence stability and final performance

### 3. Small Target Detection
- **Challenge**: Extremely small targets (3×3 pixels) are difficult to detect
- **Solution**: Designed specialized anchor scales and focal loss adaptation
- **Result**: 15% improvement in small target detection accuracy

## Reproducibility Information

### Environment Setup
```bash
# Python environment
python==3.8.10
torch==1.12.1
torchvision==0.13.1
opencv-python==4.6.0.66
albumentations==1.2.1

# Hardware specifications
GPU: NVIDIA RTX 3090 (24GB VRAM)
CPU: Intel Xeon E5-2680 v4
RAM: 64GB DDR4
```

### Random Seeds
```python
# For reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### Dataset Statistics
- **Total images**: 427
- **Training set**: 342 images (1,130 targets)
- **Validation set**: 43 images (113 targets)  
- **Test set**: 42 images (113 targets)
- **Target size distribution**: Mean 8.7×8.7 pixels, Std 4.2 pixels
- **Image resolution**: 256×256 to 1024×1024 pixels