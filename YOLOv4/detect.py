#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测推理脚本
"""

import os
import sys
import yaml
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.backbone import InfraredDetectionModel
from utils.utils import non_max_suppression, scale_coords

class InfraredDetector:
    """红外弱小目标检测器"""
    
    def __init__(self, model_path, config_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 检测参数
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
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path
        
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 确保是单通道
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 保存原始尺寸
        orig_h, orig_w = image.shape
        
        # 调整尺寸
        image_resized = cv2.resize(image, self.img_size)
        
        # 归一化
        image_normalized = (image_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        
        return image_tensor, (orig_h, orig_w)
    
    def detect(self, image_path):
        """检测图像中的目标"""
        # 预处理
        image_tensor, orig_shape = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(image_tensor)
            inference_time = time.time() - start_time
        
        # 后处理
        detections = self.postprocess(predictions, orig_shape)
        
        return detections, inference_time
    
    def postprocess(self, predictions, orig_shape):
        """后处理检测结果"""
        # 获取Transformer检测头的输出
        transformer_bbox, transformer_cls, transformer_conf = predictions['transformer']
        
        # 获取YOLO检测头的输出
        yolo_outputs = predictions['yolo']
        
        # 处理Transformer检测结果
        transformer_detections = self.process_transformer_output(
            transformer_bbox, transformer_cls, transformer_conf, orig_shape
        )
        
        # 处理YOLO检测结果
        yolo_detections = self.process_yolo_output(yolo_outputs, orig_shape)
        
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
    
    def process_transformer_output(self, bbox_pred, cls_pred, conf_pred, orig_shape):
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
                
                # 缩放到原始图像尺寸
                bbox_scaled = torch.stack([x1, y1, x2, y2], dim=1)
                bbox_scaled = scale_coords(self.img_size, bbox_scaled, orig_shape)
                
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
    
    def process_yolo_output(self, yolo_outputs, orig_shape):
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
                        
                        # 缩放到原始图像尺寸
                        bbox_scaled = torch.tensor([x1, y1, x2, y2])
                        bbox_scaled = scale_coords(self.img_size, bbox_scaled.unsqueeze(0), orig_shape).squeeze()
                        
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
    
    def visualize_detections(self, image_path, detections, save_path=None):
        """可视化检测结果"""
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path
        
        # 转换为彩色图像用于绘制
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        # 绘制检测框
        if len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                
                # 转换为整数坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 绘制边界框
                cv2.rectangle(image_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                label = f'Target: {conf:.2f}'
                cv2.putText(image_color, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, image_color)
            print(f'检测结果已保存到: {save_path}')
        
        return image_color

def main():
    parser = argparse.ArgumentParser(description='红外弱小目标检测')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='data/infrared_config.yaml', help='配置文件路径')
    parser.add_argument('--source', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='inference', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = InfraredDetector(args.model, args.config, args.device)
    
    # 更新检测参数
    detector.conf_thres = args.conf_thres
    detector.iou_thres = args.iou_thres
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理输入
    if os.path.isfile(args.source):
        # 单张图像
        image_files = [args.source]
    elif os.path.isdir(args.source):
        # 目录
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(args.source).glob(f'*{ext}'))
            image_files.extend(Path(args.source).glob(f'*{ext.upper()}'))
    else:
        print(f'错误: 输入路径不存在 {args.source}')
        return
    
    print(f'找到 {len(image_files)} 张图像')
    
    # 处理每张图像
    total_time = 0
    for i, image_file in enumerate(image_files):
        print(f'处理图像 {i+1}/{len(image_files)}: {image_file}')
        
        try:
            # 检测
            detections, inference_time = detector.detect(str(image_file))
            total_time += inference_time
            
            # 可视化
            output_path = os.path.join(args.output, f'detected_{Path(image_file).name}')
            detector.visualize_detections(str(image_file), detections, output_path)
            
            # 打印检测结果
            if len(detections) > 0:
                print(f'  检测到 {len(detections)} 个目标')
                for j, det in enumerate(detections):
                    x1, y1, x2, y2, conf, cls = det
                    print(f'    目标 {j+1}: 位置({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), 置信度: {conf:.3f}')
            else:
                print('  未检测到目标')
            
            print(f'  推理时间: {inference_time:.3f}s')
            
        except Exception as e:
            print(f'  处理图像时出错: {e}')
            continue
    
    print(f'\n检测完成！')
    print(f'总处理时间: {total_time:.3f}s')
    print(f'平均推理时间: {total_time/len(image_files):.3f}s')
    print(f'结果保存在: {args.output}')

if __name__ == '__main__':
    main()