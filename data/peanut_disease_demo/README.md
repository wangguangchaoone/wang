# 花生叶片病害演示数据集

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
