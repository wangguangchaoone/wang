"""
YOLO-MHAF训练脚本
用于训练花生叶片病害检测模型
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo_mhaf_simple import create_yolo_mhaf_n, create_yolo_mhaf_s, create_yolo_mhaf_m
from utils.datasets import PeanutDiseaseDataset, create_dataloader
from utils.metrics import compute_ap, compute_metrics
from models.losses import YOLOLoss


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class Trainer:
    """YOLO-MHAF训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging(args.log_dir)
        
        # 创建模型
        self.model = self.create_model()
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = YOLOLoss(nc=args.nc)
        
        # 优化器
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # 数据加载器
        self.train_loader = self.create_dataloader('train')
        self.val_loader = self.create_dataloader('val')
        
        # 训练状态
        self.best_map = 0.0
        self.start_epoch = 0
        
        # 恢复训练
        if args.resume:
            self.load_checkpoint(args.resume)
            
    def create_model(self):
        """创建模型"""
        if self.args.model == 'yolo_mhaf_n':
            model = create_yolo_mhaf_n(nc=self.args.nc)
        elif self.args.model == 'yolo_mhaf_s':
            model = create_yolo_mhaf_s(nc=self.args.nc)
        elif self.args.model == 'yolo_mhaf_m':
            model = create_yolo_mhaf_m(nc=self.args.nc)
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")
            
        # 打印模型信息
        info = model.get_model_info()
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Parameters: {info['total_params']:,}")
        self.logger.info(f"Model size: {info['model_size_mb']:.2f} MB")
        
        return model
        
    def create_optimizer(self):
        """创建优化器"""
        # 分离权重衰减参数
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(pg0, lr=self.args.lr, betas=(0.937, 0.999))
        elif self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(pg0, lr=self.args.lr, betas=(0.937, 0.999), weight_decay=0.0)
        else:  # SGD
            optimizer = optim.SGD(pg0, lr=self.args.lr, momentum=0.937, nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': self.args.weight_decay})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        
        self.logger.info(f'Optimizer groups: {len(pg2)} bias, {len(pg1)} weight (decay), {len(pg0)} other')
        return optimizer
        
    def create_scheduler(self):
        """创建学习率调度器"""
        if self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.epochs // 3, gamma=0.1
            )
        else:  # linear
            lf = lambda x: (1 - x / self.args.epochs) * (1.0 - 0.01) + 0.01
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
            
        return scheduler
        
    def create_dataloader(self, mode):
        """创建数据加载器"""
        return create_dataloader(
            path=self.args.data_path,
            mode=mode,
            img_size=self.args.img_size,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            augment=(mode == 'train')
        )
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for i, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True).float() / 255.0
            targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新损失
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 记录日志
            if i % self.args.log_interval == 0:
                self.logger.info(
                    f'Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item():.4f}, '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                )
                
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
        
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device, non_blocking=True).float() / 255.0
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测和目标
                all_predictions.extend(outputs)
                all_targets.extend(targets)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算指标
        metrics = compute_metrics(all_predictions, all_targets, self.args.nc)
        
        self.logger.info(f'Validation - Epoch: {epoch+1}')
        self.logger.info(f'Val Loss: {avg_loss:.4f}')
        self.logger.info(f'Precision: {metrics["precision"]:.4f}')
        self.logger.info(f'Recall: {metrics["recall"]:.4f}')
        self.logger.info(f'mAP@0.5: {metrics["map50"]:.4f}')
        self.logger.info(f'mAP@0.5:0.95: {metrics["map"]:.4f}')
        
        return avg_loss, metrics
        
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'metrics': metrics,
            'args': self.args
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.args.save_dir, 'last.pt'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'best.pt'))
            self.logger.info(f'New best model saved with mAP: {metrics["map"]:.4f}')
            
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_map = checkpoint['best_map']
            self.logger.info(f'Loaded checkpoint from {checkpoint_path} (epoch {self.start_epoch})')
        else:
            self.logger.error(f'No checkpoint found at {checkpoint_path}')
            
    def train(self):
        """主训练循环"""
        self.logger.info(f'Starting training for {self.args.epochs} epochs')
        self.logger.info(f'Device: {self.device}')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_loss, metrics = self.validate(epoch)
                
                # 更新学习率
                self.scheduler.step()
                
                # 保存检查点
                is_best = metrics['map'] > self.best_map
                if is_best:
                    self.best_map = metrics['map']
                    
                self.save_checkpoint(epoch, metrics, is_best)
                
        self.logger.info('Training completed!')
        self.logger.info(f'Best mAP: {self.best_map:.4f}')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO-MHAF Training')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolo_mhaf_n', 
                       choices=['yolo_mhaf_n', 'yolo_mhaf_s', 'yolo_mhaf_m'],
                       help='Model architecture')
    parser.add_argument('--nc', type=int, default=5, help='Number of classes')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True, help='Dataset path')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                       choices=['SGD', 'Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'linear'], help='Scheduler')
    
    # 其他参数
    parser.add_argument('--save-dir', type=str, default='./runs/train', help='Save directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--val-interval', type=int, default=5, help='Validation interval')
    parser.add_argument('--log-interval', type=int, default=50, help='Log interval')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    # 开始训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()