"""
花生叶片病害数据集处理
包含数据加载、预处理、增强等功能
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET


class PeanutDiseaseDataset(Dataset):
    """花生叶片病害数据集"""
    
    def __init__(self, data_path, mode='train', img_size=640, augment=True):
        self.data_path = Path(data_path)
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        
        # 类别映射
        self.class_names = [
            'Healthy',           # 健康
            'Early_Blight',      # 早疫病
            'Late_Blight',       # 晚疫病
            'Leaf_Spot',         # 叶斑病
            'Rust'              # 锈病
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 加载数据
        self.images, self.labels = self.load_data()
        
        # 数据增强
        self.transforms = self.get_transforms()
        
    def load_data(self):
        """加载数据"""
        images = []
        labels = []
        
        # 图像和标签路径
        img_dir = self.data_path / 'images' / self.mode
        label_dir = self.data_path / 'labels' / self.mode
        
        if not img_dir.exists() or not label_dir.exists():
            raise ValueError(f"Data directory not found: {img_dir} or {label_dir}")
        
        # 获取所有图像文件
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(img_dir.glob(ext))
        
        for img_file in img_files:
            # 查找对应的标签文件
            label_file = label_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                images.append(str(img_file))
                labels.append(str(label_file))
        
        print(f"Loaded {len(images)} {self.mode} images")
        return images, labels
    
    def get_transforms(self):
        """获取数据变换"""
        if self.augment:
            # 训练时的数据增强
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # 验证/测试时只做基本变换
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transforms
    
    def load_yolo_labels(self, label_path):
        """加载YOLO格式标签"""
        boxes = []
        class_labels = []
        
        if os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        boxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        return np.array(boxes), np.array(class_labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标签
        label_path = self.labels[idx]
        boxes, class_labels = self.load_yolo_labels(label_path)
        
        # 应用变换
        if len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            class_labels = np.array(transformed['class_labels'])
        else:
            # 没有标注框时
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = np.array([])
            class_labels = np.array([])
        
        # 构建目标tensor
        if len(boxes) > 0:
            targets = np.column_stack((
                np.zeros(len(boxes)),  # batch_idx (在collate_fn中设置)
                class_labels,
                boxes
            ))
        else:
            targets = np.zeros((0, 6))
        
        return image, torch.from_numpy(targets).float()


def collate_fn(batch):
    """自定义batch整理函数"""
    images, targets = zip(*batch)
    
    # 堆叠图像
    images = torch.stack(images, 0)
    
    # 处理targets
    for i, target in enumerate(targets):
        target[:, 0] = i  # 设置batch索引
    
    targets = torch.cat(targets, 0)
    
    return images, targets


def create_dataloader(path, mode='train', img_size=640, batch_size=16, workers=4, augment=True):
    """创建数据加载器"""
    dataset = PeanutDiseaseDataset(
        data_path=path,
        mode=mode,
        img_size=img_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(mode == 'train')
    )
    
    return dataloader


class COCODataset(Dataset):
    """COCO格式数据集"""
    
    def __init__(self, data_path, annotation_file, mode='train', img_size=640, augment=True):
        self.data_path = Path(data_path)
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        
        # 加载COCO注释
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        
        # 构建类别映射
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}
        self.class_names = list(self.categories.values())
        
        # 构建图像和注释映射
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        self.transforms = self.get_transforms()
    
    def get_transforms(self):
        """获取数据变换"""
        if self.augment:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        else:
            transforms = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        return transforms
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # 加载图像
        img_path = self.data_path / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载注释
        anns = self.annotations.get(img_id, [])
        boxes = []
        class_labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, w, h])
            class_labels.append(ann['category_id'] - 1)  # COCO类别从1开始
        
        # 应用变换
        if len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = []
            class_labels = []
        
        # 转换为YOLO格式
        if len(boxes) > 0:
            boxes = np.array(boxes)
            # COCO (x,y,w,h) -> YOLO (x_center, y_center, w, h) normalized
            boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / self.img_size
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / self.img_size
            boxes[:, 2] = boxes[:, 2] / self.img_size
            boxes[:, 3] = boxes[:, 3] / self.img_size
            
            targets = np.column_stack((
                np.zeros(len(boxes)),  # batch_idx
                class_labels,
                boxes
            ))
        else:
            targets = np.zeros((0, 6))
        
        return image, torch.from_numpy(targets).float()


def create_dataset_yaml(data_path, class_names, save_path='dataset.yaml'):
    """创建数据集配置文件"""
    config = {
        'path': str(data_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    import yaml
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Dataset config saved to {save_path}")


if __name__ == '__main__':
    # 测试数据集
    data_path = './data/peanut_disease'
    
    # 创建数据集配置
    class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Spot', 'Rust']
    create_dataset_yaml(data_path, class_names, 'peanut_disease.yaml')
    
    # 测试数据加载
    try:
        train_loader = create_dataloader(data_path, 'train', batch_size=4)
        print(f"Train dataset: {len(train_loader.dataset)} samples")
        
        # 测试一个batch
        images, targets = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch targets shape: {targets.shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is organized as:")
        print("data/peanut_disease/")
        print("├── images/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("└── labels/")
        print("    ├── train/")
        print("    ├── val/")
        print("    └── test/")