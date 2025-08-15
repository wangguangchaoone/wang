#!/usr/bin/env python3
"""
Infrared Weak and Small Target Detection Training Script
Author: Research Team
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import random
from tqdm import tqdm
import logging
import argparse
from pathlib import Path

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ModifiedResNet50(nn.Module):
    """Modified ResNet-50 backbone for infrared images"""
    def __init__(self, in_channels=1):
        super(ModifiedResNet50, self).__init__()
        
        # Import ResNet-50
        import torchvision.models as models
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

class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion module"""
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
        for i, (feat, conv) in enumerate(zip(features, self.conv_reductions)):
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

class TransformerDetectionHead(nn.Module):
    """Transformer-based detection head"""
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

class InfraredTargetDetector(nn.Module):
    """Complete infrared target detection model"""
    def __init__(self, in_channels=1, fusion_dim=256, d_model=256, 
                 nhead=8, num_layers=6):
        super(InfraredTargetDetector, self).__init__()
        
        self.backbone = ModifiedResNet50(in_channels)
        self.feature_fusion = MultiScaleFeatureFusion(fusion_dim=fusion_dim)
        self.detection_head = TransformerDetectionHead(d_model, nhead, num_layers)
        
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Fuse features
        fused_features = self.feature_fusion(features)
        
        # Detection
        cls_output, reg_output = self.detection_head(fused_features)
        
        return cls_output, reg_output

class InfraredTargetDataset(torch.utils.data.Dataset):
    """Custom dataset for infrared target detection"""
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load image and annotation paths
        self.images = list(self.data_dir.glob(f"{split}/*.png"))
        self.annotations = list(self.data_dir.glob(f"{split}/*.txt"))
        
        logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = torch.load(image_path) if image_path.suffix == '.pt' else \
               torch.from_numpy(np.load(image_path))
        
        # Load annotations
        annotation_path = self.annotations[idx]
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        # Parse annotations (format: class_id x_center y_center width height)
        targets = []
        for line in lines:
            values = line.strip().split()
            if len(values) == 5:
                targets.append([float(v) for v in values])
        
        targets = torch.tensor(targets) if targets else torch.zeros((0, 5))
        
        if self.transform:
            image, targets = self.transform(image, targets)
        
        return image, targets

class InfraredTargetAugmentation:
    """Data augmentation for infrared targets"""
    def __init__(self, config):
        self.config = config
        
    def __call__(self, image, targets):
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(*self.config['rotation_range'])
            image, targets = self.rotate_image_and_targets(image, targets, angle)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.config['scale_range'])
            image, targets = self.scale_image_and_targets(image, targets, scale)
        
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
        
        return image, targets
    
    def rotate_image_and_targets(self, image, targets, angle):
        # Implementation for rotation
        # This is a simplified version - in practice, you'd need proper coordinate transformation
        return image, targets
    
    def scale_image_and_targets(self, image, targets, scale):
        # Implementation for scaling
        # This is a simplified version - in practice, you'd need proper coordinate transformation
        return image, targets

def collate_fn(batch):
    """Custom collate function for batching"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        cls_output, reg_output = model(images)
        
        # Calculate loss
        loss = criterion(cls_output, reg_output, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            cls_output, reg_output = model(images)
            
            # Calculate loss
            loss = criterion(cls_output, reg_output, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    
    return avg_loss

class DetectionLoss(nn.Module):
    """Custom loss function for target detection"""
    def __init__(self, cls_weight=1.0, reg_weight=1.0, attention_weight=0.5):
        super(DetectionLoss, self).__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.attention_weight = attention_weight
        
        # Focal loss for classification
        self.focal_loss = self.focal_loss_fn
        
        # Smooth L1 loss for regression
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
    
    def focal_loss_fn(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss implementation"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, cls_output, reg_output, targets):
        # This is a simplified loss calculation
        # In practice, you'd need proper target assignment and loss computation
        
        # Classification loss
        cls_loss = self.focal_loss(cls_output, targets[:, 0].long())
        
        # Regression loss
        reg_loss = self.reg_loss(reg_output, targets[:, 1:])
        
        # Total loss
        total_loss = (self.cls_weight * cls_loss + 
                     self.reg_weight * reg_loss)
        
        return total_loss

def main():
    parser = argparse.ArgumentParser(description='Train Infrared Target Detector')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model setup
    model = InfraredTargetDetector(
        in_channels=1,
        fusion_dim=256,
        d_model=256,
        nhead=8,
        num_layers=6
    ).to(device)
    
    # Data augmentation configuration
    aug_config = {
        'rotation_range': [-15, 15],
        'scale_range': [0.8, 1.2],
        'brightness_range': [0.8, 1.2],
        'noise_std': 0.02
    }
    
    # Dataset and dataloader setup
    train_dataset = InfraredTargetDataset(
        args.data_dir, 
        transform=InfraredTargetAugmentation(aug_config),
        split='train'
    )
    
    val_dataset = InfraredTargetDataset(
        args.data_dir,
        transform=None,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Loss function and optimizer
    criterion = DetectionLoss(
        cls_weight=1.0,
        reg_weight=1.0,
        attention_weight=0.5
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            logger.info(f"Saved best model at epoch {epoch}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()