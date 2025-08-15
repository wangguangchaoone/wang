#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测工具函数
"""

import torch
import numpy as np
import cv2
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存模型检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, filepath)
    print(f'检查点已保存到: {filepath}')

def load_checkpoint(model, optimizer, filepath, device):
    """加载模型检查点"""
    if os.path.isfile(filepath):
        print(f"加载检查点: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"检查点加载成功，epoch: {epoch}, loss: {loss:.4f}")
        return epoch, loss
    else:
        print(f"检查点文件不存在: {filepath}")
        return 0, float('inf')

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=100):
    """非极大值抑制"""
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)
    
    # 确保输入是2D张量
    if prediction.dim() == 1:
        prediction = prediction.unsqueeze(0)
    
    # 如果没有检测结果，返回空列表
    if prediction.shape[0] == 0:
        return [torch.empty((0, 6))]
    
    # 获取置信度
    if prediction.shape[1] >= 5:
        conf = prediction[:, 4]
    else:
        conf = torch.ones(prediction.shape[0])
    
    # 筛选高置信度的检测
    mask = conf > conf_thres
    prediction = prediction[mask]
    
    if prediction.shape[0] == 0:
        return [torch.empty((0, 6))]
    
    # 获取边界框坐标
    if prediction.shape[1] >= 4:
        boxes = prediction[:, :4]
    else:
        return [torch.empty((0, 6))]
    
    # 计算边界框面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按置信度排序
    _, order = torch.sort(conf[mask], descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0]
        keep.append(i)
        
        # 计算IoU
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        ovr = inter / (area[i] + area[order[1:]] - inter)
        
        # 筛选IoU小于阈值的检测
        ids = torch.where(ovr <= iou_thres)[0]
        order = order[ids + 1]
    
    # 限制最大检测数量
    if len(keep) > max_det:
        keep = keep[:max_det]
    
    return [prediction[keep]]

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """将坐标从img1_shape缩放到img0_shape"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    
    # 裁剪到图像边界
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    
    return coords

def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 边界框格式: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_ap(recalls, precisions):
    """计算平均精度(AP)"""
    # 使用插值方法计算AP
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # 计算PR曲线下的面积
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # 计算AP
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def visualize_detections(image, detections, class_names=None, save_path=None):
    """可视化检测结果"""
    if isinstance(image, str):
        image = cv2.imread(image)
    
    if image is None:
        print("无法读取图像")
        return None
    
    # 转换为RGB格式
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # 绘制检测框
    if len(detections) > 0:
        for i, detection in enumerate(detections):
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls = detection
                
                # 绘制边界框
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # 绘制标签
                class_name = class_names[int(cls)] if class_names and int(cls) < len(class_names) else f'Class {int(cls)}'
                label = f'{class_name}: {conf:.2f}'
                ax.text(x1, y1 - 10, label, color='red', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('红外弱小目标检测结果')
    ax.axis('off')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'检测结果已保存到: {save_path}')
    
    plt.show()
    return fig

def create_heatmap(image, detections, save_path=None):
    """创建检测热力图"""
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("无法读取图像")
        return None
    
    # 创建热力图
    heatmap = np.zeros_like(image, dtype=np.float32)
    
    if len(detections) > 0:
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls = detection
                
                # 在检测区域添加高斯分布
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                sigma = max(5, int((x2 - x1 + y2 - y1) / 4))
                
                # 创建高斯核
                y, x = np.ogrid[:image.shape[0], :image.shape[1]]
                gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
                gaussian = gaussian * conf  # 根据置信度调整强度
                
                heatmap += gaussian
    
    # 归一化热力图
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title('原始红外图像')
    ax1.axis('off')
    
    # 热力图
    im = ax2.imshow(heatmap, cmap='hot', alpha=0.8)
    ax2.set_title('检测热力图')
    ax2.axis('off')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'热力图已保存到: {save_path}')
    
    plt.show()
    return fig

def calculate_metrics(detections, ground_truth, iou_threshold=0.5):
    """计算检测指标"""
    if len(detections) == 0 and len(ground_truth) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'ap': 1.0}
    
    if len(detections) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ap': 0.0}
    
    if len(ground_truth) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ap': 0.0}
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(detections), len(ground_truth)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truth):
            if len(det) >= 4 and len(gt) >= 4:
                iou_matrix[i, j] = compute_iou(det[:4], gt[:4])
    
    # 匹配检测和真实标签
    matched_detections = set()
    matched_ground_truth = set()
    
    # 按置信度排序检测结果
    if len(detections[0]) >= 5:
        sorted_indices = np.argsort([det[4] for det in detections])[::-1]
    else:
        sorted_indices = np.arange(len(detections))
    
    for det_idx in sorted_indices:
        best_gt_idx = -1
        best_iou = 0
        
        for gt_idx in range(len(ground_truth)):
            if gt_idx in matched_ground_truth:
                continue
            
            iou = iou_matrix[det_idx, gt_idx]
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matched_detections.add(det_idx)
            matched_ground_truth.add(best_gt_idx)
    
    # 计算指标
    tp = len(matched_detections)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算AP
    ap = precision * recall if precision > 0 and recall > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def time_synchronized():
    """获取同步时间"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def check_requirements(requirements_file='requirements.txt'):
    """检查依赖包是否安装"""
    if not os.path.exists(requirements_file):
        print(f"依赖文件不存在: {requirements_file}")
        return False
    
    missing_packages = []
    
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                package = line.split('>=')[0].split('==')[0].split('<=')[0]
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("所有依赖包已安装")
    return True

def get_model_info(model):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024  # 假设float32
    }