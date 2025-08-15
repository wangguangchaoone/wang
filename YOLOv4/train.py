#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测模型训练脚本
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import random

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.backbone import InfraredDetectionModel
from models.head import LossFunction
from utils.dataset import create_dataloader, InfraredDataset
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    
    # 统计指标
    losses = AverageMeter()
    bbox_losses = AverageMeter()
    cls_losses = AverageMeter()
    conf_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        images = batch['images'].to(device)
        labels = batch['labels']
        
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss, loss_dict = criterion(predictions, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        bbox_losses.update(loss_dict['bbox_loss'], images.size(0))
        cls_losses.update(loss_dict['cls_loss'], images.size(0))
        conf_losses.update(loss_dict['conf_loss'], images.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'BBox': f'{bbox_losses.avg:.4f}',
            'Cls': f'{cls_losses.avg:.4f}',
            'Conf': f'{conf_losses.avg:.4f}'
        })
        
        # 每100个batch保存一次
        if batch_idx % 100 == 0:
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth'))
    
    return {
        'loss': losses.avg,
        'bbox_loss': bbox_losses.avg,
        'cls_loss': cls_losses.avg,
        'conf_loss': conf_losses.avg
    }

def validate(model, dataloader, criterion, device, epoch):
    """验证模型"""
    model.eval()
    
    losses = AverageMeter()
    bbox_losses = AverageMeter()
    cls_losses = AverageMeter()
    conf_losses = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Validation Epoch {epoch}'):
            images = batch['images'].to(device)
            labels = batch['labels']
            
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss, loss_dict = criterion(predictions, labels)
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            bbox_losses.update(loss_dict['bbox_loss'], images.size(0))
            cls_losses.update(loss_dict['cls_loss'], images.size(0))
            conf_losses.update(loss_dict['conf_loss'], images.size(0))
    
    return {
        'loss': losses.avg,
        'bbox_loss': bbox_losses.avg,
        'cls_loss': cls_losses.avg,
        'conf_loss': conf_losses.avg
    }

def main():
    parser = argparse.ArgumentParser(description='红外弱小目标检测模型训练')
    parser.add_argument('--config', type=str, default='data/infrared_config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config.update(vars(args))
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'使用设备: {device}')
    
    # 创建模型
    model = InfraredDetectionModel(
        num_classes=config['dataset']['nc'],
        d_model=256,
        n_heads=8,
        n_layers=6
    ).to(device)
    
    # 创建损失函数
    criterion = LossFunction(
        num_classes=config['dataset']['nc'],
        anchors=3
    ).to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config['train']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从epoch {start_epoch}开始训练")
        else:
            print(f"检查点文件不存在: {args.resume}")
    
    # 创建数据加载器
    train_loader = create_dataloader(
        data_path=config['dataset']['train'],
        batch_size=args.batch_size,
        img_size=config['model']['input_size'][0],
        augment=True,
        mode='train'
    )
    
    val_loader = create_dataloader(
        data_path=config['dataset']['val'],
        batch_size=args.batch_size,
        img_size=config['model']['input_size'][0],
        augment=False,
        mode='val'
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n开始训练 Epoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        writer.add_scalar('BBox_Loss/Train', train_metrics['bbox_loss'], epoch)
        writer.add_scalar('BBox_Loss/Val', val_metrics['bbox_loss'], epoch)
        writer.add_scalar('Cls_Loss/Train', train_metrics['cls_loss'], epoch)
        writer.add_scalar('Cls_Loss/Val', val_metrics['cls_loss'], epoch)
        writer.add_scalar('Conf_Loss/Train', train_metrics['conf_loss'], epoch)
        writer.add_scalar('Conf_Loss/Val', val_metrics['conf_loss'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print(f'保存最佳模型，验证损失: {best_val_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # 打印训练信息
        print(f'Epoch {epoch+1} 完成:')
        print(f'  训练损失: {train_metrics["loss"]:.4f}')
        print(f'  验证损失: {val_metrics["loss"]:.4f}')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, args.epochs-1, val_metrics['loss'],
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    print(f'\n训练完成！最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存在: {args.save_dir}')
    
    writer.close()

if __name__ == '__main__':
    main()