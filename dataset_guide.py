#!/usr/bin/env python3
"""
花生叶片病害数据集获取完整指南
不依赖外部库，纯Python实现
"""

import os
import sys
from pathlib import Path
import argparse

def show_dataset_sources():
    """显示数据集来源信息"""
    print("=" * 80)
    print("🌱 花生叶片病害数据集获取完整指南")
    print("=" * 80)
    
    sources = [
        {
            "name": "🥇 方法1: Kaggle数据集 (推荐)",
            "description": "最丰富的植物病害数据集平台",
            "links": [
                "🔗 https://www.kaggle.com/datasets/emmarex/plantdisease",
                "🔗 https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset", 
                "🔗 https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset",
                "🔗 https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset"
            ],
            "steps": [
                "1. 注册Kaggle账号 (免费)",
                "2. 搜索关键词: 'plant disease', 'peanut leaf', 'groundnut disease'",
                "3. 下载ZIP文件到 ./raw_data/",
                "4. 解压并使用 prepare_dataset.py 转换格式",
                "5. 验证: python prepare_dataset.py --validate ./data/peanut_disease"
            ],
            "advantages": [
                "✅ 数据量大，质量高",
                "✅ 多种作物，包含相似病害",
                "✅ 社区活跃，持续更新",
                "✅ 免费下载"
            ]
        },
        {
            "name": "🥈 方法2: 学术数据集",
            "description": "来自研究论文和机构的专业数据集",
            "links": [
                "🔗 PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset",
                "🔗 Mendeley Data: https://data.mendeley.com/ (搜索 'peanut leaf disease')",
                "🔗 IEEE DataPort: https://ieee-dataport.org/",
                "🔗 UC Irvine ML Repository: https://archive.ics.uci.edu/ml/"
            ],
            "steps": [
                "1. 访问学术数据库网站",
                "2. 搜索相关论文的数据集",
                "3. 下载并引用原始来源",
                "4. 转换为YOLO格式",
                "5. 按7:1.5:1.5比例分割数据"
            ],
            "advantages": [
                "✅ 数据权威性高",
                "✅ 标注质量好",
                "✅ 有学术支撑",
                "✅ 适合研究使用"
            ]
        },
        {
            "name": "🥉 方法3: 自建数据集",
            "description": "收集和标注自己的数据集",
            "tools": [
                "🛠️ LabelImg: https://github.com/tzutalin/labelImg (推荐)",
                "🛠️ CVAT: https://github.com/openvinotoolkit/cvat",
                "🛠️ Roboflow: https://roboflow.com/",
                "🛠️ LabelMe: https://github.com/wkentaro/labelme"
            ],
            "steps": [
                "1. 收集花生叶片图像 (各种病害类型)",
                "2. 安装标注工具 (如LabelImg)",
                "3. 标记病害区域并分类",
                "4. 导出为YOLO格式",
                "5. 分割为train/val/test三个集合"
            ],
            "advantages": [
                "✅ 数据贴合实际应用",
                "✅ 可控制数据质量",
                "✅ 针对性强",
                "✅ 可持续扩展"
            ],
            "disadvantages": [
                "❌ 耗时较长",
                "❌ 需要专业知识",
                "❌ 标注成本高"
            ]
        },
        {
            "name": "🚀 方法4: 数据增强",
            "description": "扩充现有数据集",
            "methods": [
                "🔄 几何变换: 旋转、翻转、缩放、裁剪",
                "🎨 颜色调整: 亮度、对比度、饱和度",
                "🌫️ 噪声添加: 高斯噪声、椒盐噪声",
                "🔍 滤波处理: 模糊、锐化、边缘增强",
                "🌈 风格变换: 使用GAN生成新样本"
            ],
            "libraries": [
                "📦 Albumentations (项目已集成)",
                "📦 imgaug",
                "📦 torchvision.transforms",
                "📦 OpenCV"
            ]
        }
    ]
    
    for source in sources:
        print(f"\n{source['name']}")
        print("-" * 60)
        print(f"📝 {source['description']}")
        
        if 'links' in source:
            print("\n🔗 相关链接:")
            for link in source['links']:
                print(f"   {link}")
        
        if 'tools' in source:
            print("\n🛠️ 推荐工具:")
            for tool in source['tools']:
                print(f"   {tool}")
        
        if 'libraries' in source:
            print("\n📦 相关库:")
            for lib in source['libraries']:
                print(f"   {lib}")
        
        if 'methods' in source:
            print("\n🔧 主要方法:")
            for method in source['methods']:
                print(f"   {method}")
        
        if 'steps' in source:
            print("\n📋 操作步骤:")
            for step in source['steps']:
                print(f"   {step}")
        
        if 'advantages' in source:
            print("\n✨ 优势:")
            for adv in source['advantages']:
                print(f"   {adv}")
        
        if 'disadvantages' in source:
            print("\n⚠️ 缺点:")
            for dis in source['disadvantages']:
                print(f"   {dis}")

def show_specific_datasets():
    """显示具体可用的数据集"""
    print("\n" + "=" * 80)
    print("📊 具体可用数据集推荐")
    print("=" * 80)
    
    datasets = [
        {
            "name": "PlantVillage Dataset",
            "size": "54,000+ 图像",
            "classes": "38个类别，14种作物",
            "format": "分类格式 (需转换为检测格式)",
            "url": "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset",
            "pros": ["数据量大", "质量高", "广泛使用"],
            "cons": ["主要是分类任务", "可能没有花生专门数据"]
        },
        {
            "name": "New Plant Diseases Dataset",
            "size": "87,000+ 图像",
            "classes": "38个类别",
            "format": "分类格式",
            "url": "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
            "pros": ["数据量很大", "增强版本", "多样性好"],
            "cons": ["分类任务", "需要格式转换"]
        },
        {
            "name": "Plant Disease Recognition Dataset",
            "size": "20,000+ 图像",
            "classes": "15个类别",
            "format": "分类格式",
            "url": "https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset",
            "pros": ["适中大小", "处理过的数据", "易于使用"],
            "cons": ["数据量相对较小"]
        }
    ]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   📊 规模: {dataset['size']}")
        print(f"   🏷️ 类别: {dataset['classes']}")
        print(f"   📁 格式: {dataset['format']}")
        print(f"   🔗 链接: {dataset['url']}")
        print(f"   ✅ 优点: {', '.join(dataset['pros'])}")
        print(f"   ❌ 缺点: {', '.join(dataset['cons'])}")

def show_conversion_guide():
    """显示格式转换指南"""
    print("\n" + "=" * 80)
    print("🔄 数据格式转换指南")
    print("=" * 80)
    
    print("""
🎯 从分类数据集转换为检测数据集:

1. 分类 → 检测转换步骤:
   ├── 下载分类数据集 (如PlantVillage)
   ├── 使用标注工具重新标注边界框
   ├── 或使用预训练模型生成伪标签
   └── 转换为YOLO格式

2. 其他格式 → YOLO格式:
   ├── COCO格式: 使用脚本转换JSON标注
   ├── Pascal VOC格式: XML转换为YOLO txt
   ├── CSV格式: 解析坐标信息
   └── 自定义格式: 编写转换脚本

3. 使用项目工具:
   ```bash
   # 分割数据集
   python prepare_dataset.py --split ./raw_data ./data/peanut_disease
   
   # 验证数据集
   python prepare_dataset.py --validate ./data/peanut_disease
   ```

4. 标注质量检查:
   ├── 检查边界框是否准确
   ├── 确认类别标签正确
   ├── 验证坐标范围 (0-1)
   └── 检查文件名对应关系
""")

def show_quick_start():
    """显示快速开始指南"""
    print("\n" + "=" * 80)
    print("🚀 快速开始指南")
    print("=" * 80)
    
    print("""
📋 推荐的数据获取流程:

1️⃣ 立即可用 (5分钟):
   ```bash
   # 创建演示数据集结构
   python dataset_guide.py --demo
   # 手动添加一些图像进行测试
   ```

2️⃣ 快速获取 (30分钟):
   ├── 访问 Kaggle.com
   ├── 搜索 "plant disease dataset" 
   ├── 下载 PlantVillage 或类似数据集
   ├── 使用 prepare_dataset.py 处理
   └── 开始训练

3️⃣ 专业数据集 (2-3小时):
   ├── 下载多个相关数据集
   ├── 筛选花生相关图像
   ├── 使用LabelImg重新标注
   ├── 合并并分割数据集
   └── 验证数据质量

4️⃣ 自建数据集 (1-2周):
   ├── 收集花生叶片图像 (各种病害)
   ├── 使用专业标注工具
   ├── 建立标注规范和质量控制
   ├── 多人协作标注
   └── 持续扩展和优化

🎯 建议组合策略:
   ✅ 先用现有数据集快速验证模型
   ✅ 再收集特定场景数据进行微调
   ✅ 使用数据增强扩充训练集
   ✅ 持续收集困难样本改进模型
""")

def create_demo_dataset():
    """创建演示数据集"""
    print("=" * 60)
    print("🎯 创建演示数据集")
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
            ("healthy_002.txt", "0 0.4 0.6 0.35 0.3\n"),
            ("early_blight_001.txt", "1 0.6 0.4 0.2 0.3\n"),
            ("early_blight_002.txt", "1 0.3 0.7 0.25 0.2\n"),
            ("late_blight_001.txt", "2 0.4 0.6 0.25 0.35\n"),
            ("late_blight_002.txt", "2 0.7 0.3 0.2 0.4\n"),
            ("leaf_spot_001.txt", "3 0.3 0.3 0.2 0.2\n"),
            ("leaf_spot_002.txt", "3 0.8 0.5 0.15 0.25\n"),
            ("rust_001.txt", "4 0.7 0.7 0.15 0.25\n"),
            ("rust_002.txt", "4 0.2 0.4 0.3 0.2\n"),
        ],
        "val": [
            ("healthy_val_001.txt", "0 0.45 0.55 0.35 0.4\n"),
            ("early_blight_val_001.txt", "1 0.65 0.35 0.25 0.3\n"),
            ("late_blight_val_001.txt", "2 0.5 0.5 0.3 0.3\n"),
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
    
    # 创建详细的README
    readme_content = """# 花生叶片病害演示数据集

## 📁 目录结构
```
peanut_disease_demo/
├── images/
│   ├── train/          # 训练图像 (需要添加)
│   ├── val/            # 验证图像 (需要添加)
│   └── test/           # 测试图像 (需要添加)
└── labels/
    ├── train/          # 训练标签 (已创建)
    ├── val/            # 验证标签 (已创建)
    └── test/           # 测试标签 (已创建)
```

## 🏷️ 类别定义
- **0: Healthy** - 健康叶片
- **1: Early_Blight** - 早疫病
- **2: Late_Blight** - 晚疫病
- **3: Leaf_Spot** - 叶斑病
- **4: Rust** - 锈病

## 📊 数据统计
- 训练集: 10个标签文件
- 验证集: 3个标签文件
- 测试集: 2个标签文件
- 总计: 15个标签文件

## 🚀 使用方法

### 1. 添加图像文件
为每个标签文件添加对应的图像文件:
```
healthy_001.txt → healthy_001.jpg
early_blight_001.txt → early_blight_001.jpg
...
```

### 2. 验证数据集
```bash
python prepare_dataset.py --validate ./data/peanut_disease_demo
```

### 3. 开始训练
```bash
python train.py --data-path ./data/peanut_disease_demo --model yolo_mhaf_n
```

## 📝 标签格式说明
每行格式: `class_id x_center y_center width height`
- 所有坐标值都归一化到 0-1 范围
- x_center, y_center: 边界框中心点坐标
- width, height: 边界框宽度和高度

## ⚠️ 注意事项
1. 此演示数据集只包含标签文件
2. 您需要添加对应的图像文件才能使用
3. 确保图像和标签文件名匹配(除了扩展名)
4. 图像格式支持: JPG, JPEG, PNG, BMP

## 🔗 获取真实数据
参考 dataset_guide.py 获取真实的花生叶片病害数据集
"""
    
    with open(demo_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✅ 演示数据集已创建: {demo_dir}")
    print("📋 下一步:")
    print("   1. 添加对应的图像文件到 images/ 目录")
    print("   2. 运行: python prepare_dataset.py --validate ./data/peanut_disease_demo")
    print("   3. 开始训练: python train.py --data-path ./data/peanut_disease_demo")
    
    return demo_dir

def main():
    parser = argparse.ArgumentParser(description='花生叶片病害数据集获取指南')
    parser.add_argument('--sources', action='store_true',
                       help='显示数据集来源信息')
    parser.add_argument('--datasets', action='store_true',
                       help='显示具体可用数据集')
    parser.add_argument('--convert', action='store_true',
                       help='显示格式转换指南')
    parser.add_argument('--quick', action='store_true',
                       help='显示快速开始指南')
    parser.add_argument('--demo', action='store_true',
                       help='创建演示数据集')
    parser.add_argument('--all', action='store_true',
                       help='显示所有信息')
    
    args = parser.parse_args()
    
    if args.all:
        show_dataset_sources()
        show_specific_datasets()
        show_conversion_guide()
        show_quick_start()
        return
    
    if args.sources or (not any([args.datasets, args.convert, args.quick, args.demo])):
        show_dataset_sources()
    
    if args.datasets:
        show_specific_datasets()
    
    if args.convert:
        show_conversion_guide()
    
    if args.quick:
        show_quick_start()
    
    if args.demo:
        create_demo_dataset()

if __name__ == '__main__':
    main()