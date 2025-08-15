#!/usr/bin/env python3
"""
花生叶片病害数据集下载脚本
提供多种数据集获取方式
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import argparse
import shutil
from urllib.parse import urlparse
import json

def download_file(url, local_path, chunk_size=8192):
    """下载文件并显示进度"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        print(f"Downloading {os.path.basename(local_path)}...")
        print(f"File size: {total_size / 1024 / 1024:.1f} MB")
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Downloaded: {local_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """解压文件"""
    try:
        print(f"Extracting {os.path.basename(archive_path)}...")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"❌ Unsupported archive format: {archive_path}")
            return False
            
        print(f"✓ Extracted to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def download_plantvillage():
    """下载PlantVillage数据集"""
    print("=" * 60)
    print("PlantVillage Dataset Download")
    print("=" * 60)
    
    # 创建下载目录
    download_dir = Path("./downloads/plantvillage")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # PlantVillage数据集的GitHub链接
    datasets = {
        "PlantVillage-Dataset": {
            "url": "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
            "description": "Complete PlantVillage dataset with 54,000+ images"
        }
    }
    
    for name, info in datasets.items():
        print(f"\nDownloading {name}...")
        print(f"Description: {info['description']}")
        
        archive_path = download_dir / f"{name}.zip"
        
        if download_file(info['url'], archive_path):
            extract_path = download_dir / name
            if extract_archive(archive_path, extract_path):
                # 删除压缩文件以节省空间
                os.remove(archive_path)
                print(f"✓ {name} ready at: {extract_path}")
            else:
                print(f"❌ Failed to extract {name}")
        else:
            print(f"❌ Failed to download {name}")
    
    print(f"\n📁 All downloads saved to: {download_dir}")
    return download_dir

def download_sample_datasets():
    """下载示例数据集"""
    print("=" * 60)
    print("Sample Plant Disease Datasets")
    print("=" * 60)
    
    download_dir = Path("./downloads/samples")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # 一些可用的示例数据集
    sample_urls = [
        {
            "name": "Plant_Diseases_Detection",
            "url": "https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases/archive/refs/heads/master.zip",
            "description": "Plant disease detection samples"
        }
    ]
    
    for dataset in sample_urls:
        print(f"\nDownloading {dataset['name']}...")
        print(f"Description: {dataset['description']}")
        
        archive_path = download_dir / f"{dataset['name']}.zip"
        
        if download_file(dataset['url'], archive_path):
            extract_path = download_dir / dataset['name']
            if extract_archive(archive_path, extract_path):
                os.remove(archive_path)
                print(f"✓ {dataset['name']} ready at: {extract_path}")
    
    return download_dir

def create_demo_dataset():
    """创建演示数据集"""
    print("=" * 60)
    print("Creating Demo Dataset")
    print("=" * 60)
    
    demo_dir = Path("./data/peanut_disease_demo")
    
    # 创建目录结构
    dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_path in dirs:
        (demo_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # 创建示例标签文件
    sample_labels = {
        "train": [
            ("healthy_001.txt", "0 0.5 0.5 0.3 0.4\n"),
            ("early_blight_001.txt", "1 0.6 0.4 0.2 0.3\n"),
            ("late_blight_001.txt", "2 0.4 0.6 0.25 0.35\n"),
            ("leaf_spot_001.txt", "3 0.3 0.3 0.2 0.2\n"),
            ("rust_001.txt", "4 0.7 0.7 0.15 0.25\n"),
        ],
        "val": [
            ("healthy_val_001.txt", "0 0.45 0.55 0.35 0.4\n"),
            ("early_blight_val_001.txt", "1 0.65 0.35 0.25 0.3\n"),
        ],
        "test": [
            ("healthy_test_001.txt", "0 0.5 0.5 0.3 0.4\n"),
            ("rust_test_001.txt", "4 0.6 0.6 0.2 0.3\n"),
        ]
    }
    
    for mode, labels in sample_labels.items():
        for filename, content in labels:
            label_path = demo_dir / "labels" / mode / filename
            with open(label_path, 'w') as f:
                f.write(content)
    
    # 创建README
    readme_content = """# Demo Dataset for YOLO-MHAF

This is a demo dataset structure for testing the YOLO-MHAF model.

## Structure
- images/: Contains image files (you need to add actual images)
- labels/: Contains YOLO format label files

## Classes
0: Healthy
1: Early_Blight
2: Late_Blight
3: Leaf_Spot
4: Rust

## Usage
1. Add your image files to the images/ directories
2. Ensure image and label filenames match (except extension)
3. Run: python train.py --data-path ./data/peanut_disease_demo

## Note
This demo only contains label files. You need to add corresponding image files.
"""
    
    with open(demo_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Demo dataset created at: {demo_dir}")
    print("⚠️  Note: Only label files created. Add your images manually.")
    
    return demo_dir

def show_dataset_sources():
    """显示数据集来源信息"""
    print("=" * 80)
    print("🌱 花生叶片病害数据集获取指南")
    print("=" * 80)
    
    sources = [
        {
            "name": "1. Kaggle数据集",
            "description": "搜索关键词: 'plant disease', 'peanut leaf', 'groundnut disease'",
            "links": [
                "https://www.kaggle.com/datasets/emmarex/plantdisease",
                "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
                "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
            ],
            "steps": [
                "注册Kaggle账号",
                "搜索相关数据集",
                "下载并解压到 ./raw_data/",
                "使用 prepare_dataset.py 转换格式"
            ]
        },
        {
            "name": "2. 学术数据集",
            "description": "来自研究论文和机构的公开数据集",
            "links": [
                "PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset",
                "Mendeley Data: 搜索 'peanut leaf disease'",
                "IEEE DataPort: 植物病害相关数据集"
            ],
            "steps": [
                "访问学术数据库",
                "搜索相关论文的数据集",
                "下载并引用原始来源",
                "转换为YOLO格式"
            ]
        },
        {
            "name": "3. 自建数据集",
            "description": "收集和标注自己的数据集",
            "tools": [
                "LabelImg: https://github.com/tzutalin/labelImg",
                "CVAT: https://github.com/openvinotoolkit/cvat",
                "Roboflow: https://roboflow.com/"
            ],
            "steps": [
                "收集花生叶片图像",
                "使用标注工具标记病害区域",
                "导出为YOLO格式",
                "分割为train/val/test"
            ]
        },
        {
            "name": "4. 数据增强",
            "description": "扩充现有数据集",
            "methods": [
                "旋转、翻转、缩放",
                "颜色调整、对比度变化",
                "添加噪声、模糊",
                "使用Albumentations库"
            ]
        }
    ]
    
    for source in sources:
        print(f"\n{source['name']}")
        print("-" * 50)
        print(f"描述: {source['description']}")
        
        if 'links' in source:
            print("链接:")
            for link in source['links']:
                print(f"  • {link}")
        
        if 'tools' in source:
            print("工具:")
            for tool in source['tools']:
                print(f"  • {tool}")
        
        if 'steps' in source:
            print("步骤:")
            for i, step in enumerate(source['steps'], 1):
                print(f"  {i}. {step}")
        
        if 'methods' in source:
            print("方法:")
            for method in source['methods']:
                print(f"  • {method}")
    
    print("\n" + "=" * 80)
    print("💡 推荐流程:")
    print("1. 先尝试下载Kaggle上的现有数据集")
    print("2. 如果数据不足，考虑数据增强")
    print("3. 最后考虑自建数据集补充特定场景")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Download peanut disease datasets')
    parser.add_argument('--plantvillage', action='store_true',
                       help='Download PlantVillage dataset')
    parser.add_argument('--samples', action='store_true',
                       help='Download sample datasets')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo dataset structure')
    parser.add_argument('--sources', action='store_true',
                       help='Show dataset sources information')
    parser.add_argument('--all', action='store_true',
                       help='Download all available datasets')
    
    args = parser.parse_args()
    
    if args.sources or (not any([args.plantvillage, args.samples, args.demo, args.all])):
        show_dataset_sources()
        return
    
    if args.all or args.plantvillage:
        try:
            download_plantvillage()
        except Exception as e:
            print(f"❌ PlantVillage download failed: {e}")
    
    if args.all or args.samples:
        try:
            download_sample_datasets()
        except Exception as e:
            print(f"❌ Sample datasets download failed: {e}")
    
    if args.all or args.demo:
        try:
            create_demo_dataset()
        except Exception as e:
            print(f"❌ Demo dataset creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("1. Check downloaded datasets in ./downloads/")
    print("2. Use prepare_dataset.py to convert formats if needed")
    print("3. Validate dataset: python prepare_dataset.py --validate")
    print("4. Start training: python train.py")
    print("=" * 60)

if __name__ == '__main__':
    main()