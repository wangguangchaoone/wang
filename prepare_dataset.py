#!/usr/bin/env python3
"""
数据集准备辅助脚本
帮助用户准备和验证花生叶片病害数据集
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
import random


def create_dataset_structure(data_path):
    """创建标准数据集目录结构"""
    data_path = Path(data_path)
    
    # 创建目录
    dirs_to_create = [
        'images/train',
        'images/val', 
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]
    
    for dir_name in dirs_to_create:
        dir_path = data_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print(f"\n✓ Dataset structure created at: {data_path}")


def validate_dataset(data_path):
    """验证数据集完整性"""
    data_path = Path(data_path)
    
    print(f"Validating dataset at: {data_path}")
    print("=" * 50)
    
    # 检查目录结构
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✓ {dir_name}/")
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False
    
    # 检查数据文件
    modes = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    class_counts = defaultdict(int)
    
    for mode in modes:
        img_dir = data_path / 'images' / mode
        label_dir = data_path / 'labels' / mode
        
        # 统计图像文件
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(img_dir.glob(ext))
        
        # 统计标签文件
        label_files = list(label_dir.glob('*.txt'))
        
        # 检查匹配
        matched_pairs = 0
        for img_file in img_files:
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                matched_pairs += 1
                
                # 统计类别
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                except:
                    pass
        
        print(f"\n{mode.upper()} set:")
        print(f"  Images: {len(img_files)}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Matched pairs: {matched_pairs}")
        
        total_images += len(img_files)
        total_labels += len(label_files)
    
    print(f"\nDataset Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    
    # 类别分布
    if class_counts:
        print(f"\nClass distribution:")
        class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Spot', 'Rust']
        for class_id in range(5):
            class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
            count = class_counts.get(class_id, 0)
            print(f"  {class_id} ({class_name}): {count} objects")
    
    if total_images == 0:
        print(f"\n⚠️  No images found! Please add your dataset files.")
        return False
    
    print(f"\n✓ Dataset validation completed!")
    return True


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """将单个目录的数据分割为train/val/test"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # 创建目标结构
    create_dataset_structure(target_dir)
    
    # 获取所有图像文件
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(source_dir.glob(f"**/{ext}"))
    
    if not img_files:
        print(f"❌ No images found in {source_dir}")
        return False
    
    # 随机打乱
    random.shuffle(img_files)
    
    # 计算分割点
    total = len(img_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': img_files[:train_end],
        'val': img_files[train_end:val_end],
        'test': img_files[val_end:]
    }
    
    # 复制文件
    for mode, files in splits.items():
        img_target = target_dir / 'images' / mode
        label_target = target_dir / 'labels' / mode
        
        copied_images = 0
        copied_labels = 0
        
        for img_file in files:
            # 复制图像
            target_img = img_target / img_file.name
            shutil.copy2(img_file, target_img)
            copied_images += 1
            
            # 查找并复制对应的标签
            label_file = img_file.parent / f"{img_file.stem}.txt"
            if label_file.exists():
                target_label = label_target / f"{img_file.stem}.txt"
                shutil.copy2(label_file, target_label)
                copied_labels += 1
        
        print(f"{mode}: {copied_images} images, {copied_labels} labels")
    
    print(f"\n✓ Dataset split completed!")
    return True


def download_sample_dataset():
    """下载示例数据集的说明"""
    print("Sample Dataset Download Guide:")
    print("=" * 50)
    print("1. PlantVillage Dataset:")
    print("   - Visit: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("   - Download and extract to ./raw_data/")
    print()
    print("2. Plant Pathology Dataset:")
    print("   - Visit: https://www.kaggle.com/c/plant-pathology-2020-fgvc7")
    print("   - Download competition data")
    print()
    print("3. Custom Dataset:")
    print("   - Use LabelImg: https://github.com/tzutalin/labelImg")
    print("   - Annotate your images in YOLO format")
    print()
    print("After downloading, use:")
    print("python prepare_dataset.py --split /path/to/raw/data ./data/peanut_disease")


def main():
    parser = argparse.ArgumentParser(description='Prepare peanut disease dataset')
    parser.add_argument('--create', type=str, help='Create dataset structure at path')
    parser.add_argument('--validate', type=str, help='Validate dataset at path') 
    parser.add_argument('--split', nargs=2, metavar=('SOURCE', 'TARGET'), 
                       help='Split dataset from source to target directory')
    parser.add_argument('--download-guide', action='store_true', 
                       help='Show dataset download guide')
    parser.add_argument('--train-ratio', type=float, default=0.7, 
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    if args.create:
        create_dataset_structure(args.create)
    elif args.validate:
        validate_dataset(args.validate)
    elif args.split:
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        split_dataset(args.split[0], args.split[1], 
                     args.train_ratio, args.val_ratio, test_ratio)
    elif args.download_guide:
        download_sample_dataset()
    else:
        print("YOLO-MHAF Dataset Preparation Tool")
        print("=" * 40)
        print("Usage examples:")
        print("  python prepare_dataset.py --create ./data/peanut_disease")
        print("  python prepare_dataset.py --validate ./data/peanut_disease") 
        print("  python prepare_dataset.py --split ./raw_data ./data/peanut_disease")
        print("  python prepare_dataset.py --download-guide")
        print()
        print("For help: python prepare_dataset.py -h")


if __name__ == '__main__':
    main()