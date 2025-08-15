#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测数据集转换与划分脚本
"""

import os
import cv2
import numpy as np
import shutil
import random
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

class InfraredDatasetConverter:
    """红外数据集转换器"""
    
    def __init__(self, input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 创建输出目录结构
        self.create_output_dirs()
        
        # 支持的图像格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 支持的标注格式
        self.annotation_extensions = {'.xml', '.json', '.txt'}
    
    def create_output_dirs(self):
        """创建输出目录结构"""
        # 创建主目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        # 创建数据列表文件
        self.train_list_file = self.output_dir / 'train.txt'
        self.val_list_file = self.output_dir / 'val.txt'
        self.test_list_file = self.output_dir / 'test.txt'
    
    def find_image_files(self):
        """查找所有图像文件"""
        image_files = []
        
        for ext in self.image_extensions:
            image_files.extend(self.input_dir.rglob(f'*{ext}'))
            image_files.extend(self.input_dir.rglob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def find_annotation_file(self, image_path):
        """查找对应的标注文件"""
        image_name = image_path.stem
        
        # 尝试不同的标注格式
        for ext in self.annotation_extensions:
            annotation_path = image_path.parent / f'{image_name}{ext}'
            if annotation_path.exists():
                return annotation_path
        
        # 如果没找到，返回None
        return None
    
    def convert_voc_to_yolo(self, xml_path, image_shape):
        """将VOC格式转换为YOLO格式"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            # 获取类别
            class_name = obj.find('name').text
            class_id = self.get_class_id(class_name)
            
            # 获取边界框
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 转换为YOLO格式 (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # 确保坐标在[0, 1]范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append([class_id, x_center, y_center, width, height])
        
        return yolo_annotations
    
    def convert_coco_to_yolo(self, json_path, image_name):
        """将COCO格式转换为YOLO格式"""
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 获取图像信息
        image_info = None
        for img in coco_data['images']:
            if img['file_name'] == image_name:
                image_info = img
                break
        
        if image_info is None:
            return []
        
        img_width = image_info['width']
        img_height = image_info['height']
        
        yolo_annotations = []
        
        # 获取该图像的所有标注
        image_id = image_info['id']
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                # 获取类别
                class_id = ann['category_id'] - 1  # COCO类别ID从1开始
                
                # 获取边界框
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 转换为YOLO格式
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # 确保坐标在[0, 1]范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_annotations.append([class_id, x_center, y_center, width, height])
        
        return yolo_annotations
    
    def convert_txt_to_yolo(self, txt_path):
        """将TXT格式转换为YOLO格式（如果已经是YOLO格式则直接返回）"""
        yolo_annotations = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
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
                            
                            # 确保坐标在[0, 1]范围内
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            yolo_annotations.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            print(f"读取TXT文件出错 {txt_path}: {e}")
        
        return yolo_annotations
    
    def get_class_id(self, class_name):
        """获取类别ID（这里简化为红外弱小目标类别）"""
        # 可以根据实际需求修改类别映射
        class_mapping = {
            'infrared_target': 0,
            'target': 0,
            'object': 0,
            'small_target': 0,
            'weak_target': 0
        }
        
        return class_mapping.get(class_name.lower(), 0)
    
    def save_yolo_annotation(self, annotations, output_path):
        """保存YOLO格式的标注文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    def process_dataset(self):
        """处理整个数据集"""
        print("开始处理数据集...")
        
        # 查找所有图像文件
        image_files = self.find_image_files()
        print(f"找到 {len(image_files)} 张图像")
        
        if len(image_files) == 0:
            print("未找到图像文件！")
            return
        
        # 随机打乱文件顺序
        random.shuffle(image_files)
        
        # 计算分割点
        total_files = len(image_files)
        train_end = int(total_files * self.train_ratio)
        val_end = train_end + int(total_files * self.val_ratio)
        
        # 分割数据集
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        print(f"数据集分割: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")
        
        # 处理各个数据集
        self.process_split(train_files, 'train')
        self.process_split(val_files, 'val')
        self.process_split(test_files, 'test')
        
        # 创建数据列表文件
        self.create_data_lists()
        
        print("数据集处理完成！")
    
    def process_split(self, image_files, split_name):
        """处理特定分割的数据集"""
        print(f"处理 {split_name} 集...")
        
        for image_path in tqdm(image_files, desc=f"处理 {split_name} 集"):
            try:
                # 查找标注文件
                annotation_path = self.find_annotation_file(image_path)
                
                # 转换标注格式
                yolo_annotations = []
                if annotation_path is not None:
                    if annotation_path.suffix.lower() == '.xml':
                        # VOC格式
                        yolo_annotations = self.convert_voc_to_yolo(annotation_path, None)
                    elif annotation_path.suffix.lower() == '.json':
                        # COCO格式
                        yolo_annotations = self.convert_coco_to_yolo(annotation_path, image_path.name)
                    elif annotation_path.suffix.lower() == '.txt':
                        # TXT格式（可能是YOLO格式）
                        yolo_annotations = self.convert_txt_to_yolo(annotation_path)
                
                # 如果没有找到标注文件，创建空的标注文件
                if len(yolo_annotations) == 0:
                    print(f"警告: 图像 {image_path.name} 没有对应的标注文件")
                
                # 复制图像文件
                output_image_path = self.output_dir / 'images' / split_name / image_path.name
                shutil.copy2(image_path, output_image_path)
                
                # 保存YOLO标注文件
                output_label_path = self.output_dir / 'labels' / split_name / f'{image_path.stem}.txt'
                self.save_yolo_annotation(yolo_annotations, output_label_path)
                
            except Exception as e:
                print(f"处理文件 {image_path} 时出错: {e}")
                continue
    
    def create_data_lists(self):
        """创建数据列表文件"""
        # 训练集列表
        with open(self.train_list_file, 'w', encoding='utf-8') as f:
            for image_path in (self.output_dir / 'images' / 'train').glob('*'):
                if image_path.suffix.lower() in self.image_extensions:
                    f.write(f"{image_path.absolute()}\n")
        
        # 验证集列表
        with open(self.val_list_file, 'w', encoding='utf-8') as f:
            for image_path in (self.output_dir / 'images' / 'val').glob('*'):
                if image_path.suffix.lower() in self.image_extensions:
                    f.write(f"{image_path.absolute()}\n")
        
        # 测试集列表
        with open(self.test_list_file, 'w', encoding='utf-8') as f:
            for image_path in (self.output_dir / 'images' / 'test').glob('*'):
                if image_path.suffix.lower() in self.image_extensions:
                    f.write(f"{image_path.absolute()}\n")
        
        print(f"数据列表文件已创建:")
        print(f"  训练集: {self.train_list_file}")
        print(f"  验证集: {self.val_list_file}")
        print(f"  测试集: {self.test_list_file}")

def create_sample_dataset(output_dir, num_samples=100):
    """创建示例数据集（用于测试）"""
    print(f"创建示例数据集，包含 {num_samples} 张图像...")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图像和标注目录
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'labels').mkdir(exist_ok=True)
    
    for i in range(num_samples):
        # 创建随机图像
        img_size = 512
        image = np.random.randint(0, 255, (img_size, img_size), dtype=np.uint8)
        
        # 添加一些随机噪声和目标
        for _ in range(random.randint(1, 5)):
            # 随机目标位置和大小
            x = random.randint(50, img_size-50)
            y = random.randint(50, img_size-50)
            size = random.randint(10, 30)
            
            # 绘制目标（简单的矩形）
            cv2.rectangle(image, (x-size//2, y-size//2), (x+size//2, y+size//2), 255, -1)
        
        # 保存图像
        image_path = output_dir / 'images' / f'sample_{i:04d}.jpg'
        cv2.imwrite(str(image_path), image)
        
        # 创建对应的标注文件
        label_path = output_dir / 'labels' / f'sample_{i:04d}.txt'
        
        # 生成随机标注（YOLO格式）
        with open(label_path, 'w') as f:
            for _ in range(random.randint(1, 3)):
                # 随机目标位置和大小
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)
                width = random.uniform(0.05, 0.15)
                height = random.uniform(0.05, 0.15)
                
                # 确保边界框在图像内
                x_center = max(width/2, min(1-width/2, x_center))
                y_center = max(height/2, min(1-height/2, y_center))
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"示例数据集已创建在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='红外弱小目标检测数据集转换与划分')
    parser.add_argument('--input', type=str, help='输入数据集目录')
    parser.add_argument('--output', type=str, default='VOCdevkit', help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--create-sample', action='store_true', help='创建示例数据集')
    parser.add_argument('--sample-size', type=int, default=100, help='示例数据集大小')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    if args.create_sample:
        # 创建示例数据集
        create_sample_dataset(args.output, args.sample_size)
    elif args.input:
        # 转换现有数据集
        if not os.path.exists(args.input):
            print(f"错误: 输入目录不存在 {args.input}")
            return
        
        # 验证比例
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"警告: 比例总和为 {total_ratio}，不等于1.0")
            # 重新归一化
            args.train_ratio /= total_ratio
            args.val_ratio /= total_ratio
            args.test_ratio /= total_ratio
            print(f"已重新归一化: 训练集 {args.train_ratio:.3f}, 验证集 {args.val_ratio:.3f}, 测试集 {args.test_ratio:.3f}")
        
        # 创建转换器
        converter = InfraredDatasetConverter(
            args.input,
            args.output,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
        
        # 处理数据集
        converter.process_dataset()
    else:
        print("请指定输入目录或使用 --create-sample 创建示例数据集")
        print("\n使用示例:")
        print("  转换现有数据集:")
        print("    python Dataset_partitioning.py --input /path/to/dataset --output VOCdevkit")
        print("  创建示例数据集:")
        print("    python Dataset_partitioning.py --create-sample --output VOCdevkit")

if __name__ == '__main__':
    main()