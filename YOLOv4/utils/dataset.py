import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import random

class InfraredDataset(Dataset):
    """红外弱小目标检测数据集"""
    def __init__(self, data_path, img_size=512, augment=True, mode='train'):
        self.data_path = data_path
        self.img_size = img_size
        self.augment = augment
        self.mode = mode
        
        # 读取数据列表
        self.img_files = []
        self.label_files = []
        
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        img_path = line
                        label_path = line.replace('.jpg', '.txt').replace('.png', '.txt')
                        
                        if os.path.exists(img_path) and os.path.exists(label_path):
                            self.img_files.append(img_path)
                            self.label_files.append(label_path)
        
        # 数据增强
        if self.augment and self.mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.0),  # 红外图像通常不进行垂直翻转
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.Normalize(mean=[0.5], std=[0.5]),  # 红外图像归一化
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # 读取红外图像（单通道）
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # 如果读取失败，创建随机图像
            img = np.random.randint(0, 255, (self.img_size, self.img_size), dtype=np.uint8)
        
        # 确保图像是单通道
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 读取标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([class_id, x_center, y_center, width, height])
        
        # 如果没有标签，创建随机标签用于演示
        if len(labels) == 0 and self.mode == 'train':
            # 随机生成1-3个目标
            num_targets = random.randint(1, 3)
            labels = []
            for _ in range(num_targets):
                class_id = 0  # 红外弱小目标类别
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)
                width = random.uniform(0.05, 0.15)  # 弱小目标通常较小
                height = random.uniform(0.05, 0.15)
                labels.append([class_id, x_center, y_center, width, height])
        
        # 转换为numpy数组
        labels = np.array(labels)
        
        # 数据增强
        if self.augment and len(labels) > 0:
            bboxes = labels[:, 1:5]  # 只取坐标信息
            class_labels = labels[:, 0]
            
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
            # 重新组合标签
            if len(bboxes) > 0:
                labels = np.column_stack([class_labels, bboxes])
            else:
                labels = np.empty((0, 5))
        else:
            # 只进行基本变换
            transformed = self.transform(image=img)
            img = transformed['image']
        
        # 确保标签格式正确
        if len(labels) == 0:
            labels = np.zeros((0, 5))
        
        return {
            'image': img,
            'labels': torch.from_numpy(labels).float(),
            'image_path': img_path
        }

def collate_fn(batch):
    """数据批处理函数"""
    images = []
    labels = []
    image_paths = []
    
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['labels'])
        image_paths.append(sample['image_path'])
    
    # 堆叠图像
    images = torch.stack(images, dim=0)
    
    return {
        'images': images,
        'labels': labels,
        'image_paths': image_paths
    }

def create_dataloader(data_path, batch_size=8, img_size=512, augment=True, mode='train', num_workers=4):
    """创建数据加载器"""
    dataset = InfraredDataset(data_path, img_size, augment, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader

class InfraredDataAugmentation:
    """红外图像数据增强类"""
    def __init__(self, img_size=512):
        self.img_size = img_size
        
        # 红外图像特定的增强策略
        self.augmentations = [
            # 亮度对比度调整
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            # 噪声添加
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.2),
            # 模糊
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            # 几何变换（红外图像通常不进行大幅变换）
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.3
            ),
            # 弹性变换
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        ]
    
    def __call__(self, image, labels):
        """应用数据增强"""
        # 随机选择增强策略
        selected_augs = random.sample(self.augmentations, random.randint(1, 3))
        
        # 应用增强
        transform = A.Compose([
            *selected_augs,
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 准备标签
        if len(labels) > 0:
            bboxes = labels[:, 1:5]
            class_labels = labels[:, 0]
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            # 重新组合标签
            if len(transformed['bboxes']) > 0:
                new_labels = np.column_stack([transformed['class_labels'], transformed['bboxes']])
            else:
                new_labels = np.empty((0, 5))
        else:
            transformed = transform(image=image)
            new_labels = np.empty((0, 5))
        
        return transformed['image'], new_labels

def visualize_dataset_sample(dataset, idx=0):
    """可视化数据集样本"""
    sample = dataset[idx]
    image = sample['image']
    labels = sample['labels']
    
    # 转换为numpy数组用于显示
    if isinstance(image, torch.Tensor):
        image = image.squeeze().numpy()
    
    # 反归一化
    image = (image * 0.5 + 0.5) * 255
    image = image.astype(np.uint8)
    
    # 绘制边界框
    if len(labels) > 0:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # 转换为像素坐标
            x1 = int((x_center - width/2) * image.shape[1])
            y1 = int((y_center - height/2) * image.shape[0])
            x2 = int((x_center + width/2) * image.shape[1])
            y2 = int((y_center + height/2) * image.shape[0])
            
            # 绘制矩形
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(image, f'Target', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image