#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测模型验证脚本
"""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.backbone import InfraredDetectionModel
from utils.dataset import create_dataloader
from utils.utils import non_max_suppression, scale_coords, compute_ap, compute_iou

class InfraredValidator:
    """红外弱小目标检测验证器"""
    
    def __init__(self, model_path, config_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 验证参数
        self.conf_thres = self.config['detection']['conf_thres']
        self.iou_thres = self.config['detection']['iou_thres']
        self.max_det = self.config['detection']['max_det']
        self.img_size = self.config['model']['input_size']
        
        print(f'模型加载完成，使用设备: {self.device}')
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        # 创建模型
        model = InfraredDetectionModel(
            num_classes=self.config['dataset']['nc'],
            d_model=256,
            n_heads=8,
            n_layers=6
        )
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f'成功加载模型权重: {model_path}')
        else:
            print(f'警告: 模型文件不存在 {model_path}，使用随机初始化的模型')
        
        return model.to(self.device)
    
    def validate(self, val_loader):
        """验证模型性能"""
        self.model.eval()
        
        # 统计指标
        stats = []
        total_time = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='验证中')):
                images = batch['images'].to(self.device)
                labels = batch['labels']
                image_paths = batch['image_paths']
                
                # 推理
                start_time = time.time()
                predictions = self.model(images)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # 后处理
                detections = self.postprocess(predictions, images.shape[2:])
                
                # 计算每张图像的指标
                for i, (det, label, img_path) in enumerate(zip(detections, labels, image_paths)):
                    # 转换标签格式
                    if len(label) > 0:
                        # 标签格式: [class, x_center, y_center, width, height]
                        # 转换为: [x1, y1, x2, y2, class]
                        label_boxes = []
                        for l in label:
                            class_id = int(l[0])
                            x_center, y_center, width, height = l[1:5]
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            label_boxes.append([x1, y1, x2, y2, class_id])
                        label_boxes = np.array(label_boxes)
                    else:
                        label_boxes = np.empty((0, 5))
                    
                    # 转换检测结果格式
                    if len(det) > 0:
                        # 检测格式: [x1, y1, x2, y2, conf, class]
                        # 转换为: [x1, y1, x2, y2, conf, class]
                        det_boxes = det.cpu().numpy()
                        det_boxes = det_boxes[:, [0, 1, 2, 3, 4, 5]]  # 保持原格式
                    else:
                        det_boxes = np.empty((0, 6))
                    
                    # 计算指标
                    ap, precision, recall = self.compute_metrics(det_boxes, label_boxes)
                    
                    stats.append({
                        'image_path': img_path,
                        'ap': ap,
                        'precision': precision,
                        'recall': recall,
                        'num_detections': len(det_boxes),
                        'num_ground_truth': len(label_boxes),
                        'inference_time': inference_time
                    })
        
        # 计算总体指标
        overall_metrics = self.compute_overall_metrics(stats)
        
        return overall_metrics, stats
    
    def postprocess(self, predictions, img_shape):
        """后处理检测结果"""
        # 获取Transformer检测头的输出
        transformer_bbox, transformer_cls, transformer_conf = predictions['transformer']
        
        # 获取YOLO检测头的输出
        yolo_outputs = predictions['yolo']
        
        # 处理Transformer检测结果
        transformer_detections = self.process_transformer_output(
            transformer_bbox, transformer_cls, transformer_conf, img_shape
        )
        
        # 处理YOLO检测结果
        yolo_detections = self.process_yolo_output(yolo_outputs, img_shape)
        
        # 合并检测结果
        all_detections = torch.cat([transformer_detections, yolo_detections], dim=0)
        
        # 非极大值抑制
        if len(all_detections) > 0:
            all_detections = non_max_suppression(
                all_detections,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_det
            )[0]
        
        return all_detections
    
    def process_transformer_output(self, bbox_pred, cls_pred, conf_pred, img_shape):
        """处理Transformer检测头的输出"""
        batch_size = bbox_pred.size(0)
        detections = []
        
        for b in range(batch_size):
            bbox = bbox_pred[b]  # (A, 4)
            cls = cls_pred[b]    # (A, C)
            conf = conf_pred[b]  # (A, 1)
            
            # 获取每个anchor的最高置信度类别
            max_conf, max_cls = torch.max(cls, dim=1)
            
            # 筛选高置信度的检测
            mask = conf.squeeze() > self.conf_thres
            if mask.sum() > 0:
                bbox_filtered = bbox[mask]
                cls_filtered = max_cls[mask]
                conf_filtered = conf[mask]
                
                # 转换为检测格式 [x1, y1, x2, y2, conf, cls]
                x1 = bbox_filtered[:, 0] - bbox_filtered[:, 2] / 2
                y1 = bbox_filtered[:, 1] - bbox_filtered[:, 3] / 2
                x2 = bbox_filtered[:, 0] + bbox_filtered[:, 2] / 2
                y2 = bbox_filtered[:, 1] + bbox_filtered[:, 3] / 2
                
                # 缩放到图像尺寸
                bbox_scaled = torch.stack([x1, y1, x2, y2], dim=1)
                bbox_scaled = scale_coords(self.img_size, bbox_scaled, img_shape)
                
                # 组合检测结果
                detection = torch.cat([
                    bbox_scaled,
                    conf_filtered,
                    cls_filtered.unsqueeze(1).float()
                ], dim=1)
                
                detections.append(detection)
        
        if detections:
            return torch.cat(detections, dim=0)
        else:
            return torch.empty((0, 6))
    
    def process_yolo_output(self, yolo_outputs, img_shape):
        """处理YOLO检测头的输出"""
        all_detections = []
        
        for bbox_pred, cls_pred, obj_pred in yolo_outputs:
            batch_size = bbox_pred.size(0)
            
            for b in range(batch_size):
                bbox = bbox_pred[b]  # (A*4, H, W)
                cls = cls_pred[b]    # (A*C, H, W)
                obj = obj_pred[b]    # (A*1, H, W)
                
                # 重塑为 (A, H, W, 4/1/C)
                num_anchors = bbox.size(0) // 4
                bbox = bbox.view(num_anchors, 4, bbox.size(1), bbox.size(2))
                cls = cls.view(num_anchors, -1, cls.size(1), cls.size(2))
                obj = obj.view(num_anchors, 1, obj.size(1), obj.size(2))
                
                # 获取每个位置的最高置信度
                max_conf, max_cls = torch.max(cls, dim=1)
                
                # 筛选高置信度检测
                mask = obj.squeeze() > self.conf_thres
                if mask.sum() > 0:
                    # 获取检测位置
                    det_indices = torch.where(mask)
                    
                    for i in range(len(det_indices[0])):
                        anchor_idx = det_indices[0][i]
                        y = det_indices[1][i]
                        x = det_indices[2][i]
                        
                        # 获取检测结果
                        bbox_val = bbox[anchor_idx, :, y, x]
                        conf_val = obj[anchor_idx, 0, y, x]
                        cls_val = max_cls[anchor_idx, y, x]
                        
                        # 转换为绝对坐标
                        x_center = (x + bbox_val[0]) / bbox.size(2)
                        y_center = (y + bbox_val[1]) / bbox.size(1)
                        width = bbox_val[2]
                        height = bbox_val[3]
                        
                        # 转换为边界框格式
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        # 缩放到图像尺寸
                        bbox_scaled = torch.tensor([x1, y1, x2, y2])
                        bbox_scaled = scale_coords(self.img_size, bbox_scaled.unsqueeze(0), img_shape).squeeze()
                        
                        # 组合检测结果
                        detection = torch.cat([
                            bbox_scaled,
                            conf_val.unsqueeze(0),
                            cls_val.unsqueeze(0).float()
                        ])
                        
                        all_detections.append(detection)
        
        if all_detections:
            return torch.stack(all_detections, dim=0)
        else:
            return torch.empty((0, 6))
    
    def compute_metrics(self, detections, ground_truth):
        """计算单张图像的指标"""
        if len(detections) == 0 and len(ground_truth) == 0:
            return 1.0, 1.0, 1.0  # 完美匹配
        
        if len(detections) == 0:
            return 0.0, 0.0, 0.0  # 没有检测到任何目标
        
        if len(ground_truth) == 0:
            return 0.0, 0.0, 0.0  # 没有真实标签
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(detections), len(ground_truth)))
        for i, det in enumerate(detections):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = compute_iou(det[:4], gt[:4])
        
        # 匹配检测和真实标签
        matched_detections = set()
        matched_ground_truth = set()
        
        # 按置信度排序检测结果
        sorted_indices = np.argsort(detections[:, 4])[::-1]
        
        for det_idx in sorted_indices:
            best_gt_idx = -1
            best_iou = 0
            
            for gt_idx in range(len(ground_truth)):
                if gt_idx in matched_ground_truth:
                    continue
                
                iou = iou_matrix[det_idx, gt_idx]
                if iou > best_iou and iou >= self.iou_thres:
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
        
        # 计算AP (Average Precision)
        if precision > 0 and recall > 0:
            ap = precision * recall
        else:
            ap = 0.0
        
        return ap, precision, recall
    
    def compute_overall_metrics(self, stats):
        """计算总体指标"""
        if not stats:
            return {}
        
        # 计算平均值
        avg_ap = np.mean([s['ap'] for s in stats])
        avg_precision = np.mean([s['precision'] for s in stats])
        avg_recall = np.mean([s['recall'] for s in stats])
        avg_inference_time = np.mean([s['inference_time'] for s in stats])
        
        # 计算总数
        total_detections = sum([s['num_detections'] for s in stats])
        total_ground_truth = sum([s['num_ground_truth'] for s in stats])
        
        # 计算F1分数
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'mAP': avg_ap,
            'mPrecision': avg_precision,
            'mRecall': avg_recall,
            'F1_Score': f1_score,
            'avg_inference_time': avg_inference_time,
            'total_detections': total_detections,
            'total_ground_truth': total_ground_truth,
            'total_images': len(stats)
        }

def main():
    parser = argparse.ArgumentParser(description='红外弱小目标检测模型验证')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='data/infrared_config.yaml', help='配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='验证数据路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--output', type=str, default='runs/val', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建验证器
    validator = InfraredValidator(args.model, args.config, args.device)
    
    # 更新验证参数
    validator.conf_thres = args.conf_thres
    validator.iou_thres = args.iou_thres
    
    # 创建数据加载器
    val_loader = create_dataloader(
        data_path=args.data,
        batch_size=args.batch_size,
        img_size=validator.img_size[0],
        augment=False,
        mode='val'
    )
    
    print(f'开始验证，共 {len(val_loader.dataset)} 张图像')
    
    # 执行验证
    overall_metrics, detailed_stats = validator.validate(val_loader)
    
    # 打印结果
    print('\n' + '='*50)
    print('验证结果')
    print('='*50)
    print(f'mAP: {overall_metrics["mAP"]:.4f}')
    print(f'mPrecision: {overall_metrics["mPrecision"]:.4f}')
    print(f'mRecall: {overall_metrics["mRecall"]:.4f}')
    print(f'F1 Score: {overall_metrics["F1_Score"]:.4f}')
    print(f'平均推理时间: {overall_metrics["avg_inference_time"]:.4f}s')
    print(f'总检测数: {overall_metrics["total_detections"]}')
    print(f'总真实标签数: {overall_metrics["total_ground_truth"]}')
    print(f'总图像数: {overall_metrics["total_images"]}')
    
    # 保存详细结果
    import json
    results_file = os.path.join(args.output, 'validation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_metrics': overall_metrics,
            'detailed_stats': detailed_stats
        }, f, indent=2, ensure_ascii=False)
    
    print(f'\n详细结果已保存到: {results_file}')

if __name__ == '__main__':
    main()