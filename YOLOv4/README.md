# 红外弱小目标检测项目

这是一个基于深度学习的红外弱小目标检测项目，结合了卷积神经网络（CNN）和Transformer技术，专门用于检测红外图像中的弱小目标。

## 项目特点

- **混合架构**: 结合CNN骨干网络和Transformer检测头
- **多尺度检测**: 使用特征金字塔网络（FPN）进行多尺度特征融合
- **红外图像优化**: 专门针对红外图像特点进行优化
- **完整流程**: 包含训练、验证、检测的完整流程
- **易于使用**: 提供简单的命令行接口

## 目录结构

```
YOLOv4/
├── data/                   # 数据相关配置文件
│   └── infrared_config.yaml
├── images/                 # 测试图片文件夹
├── inference/              # 推理结果保存文件夹
├── models/                 # 模型相关配置文件和对应代码
│   ├── backbone.py        # 骨干网络
│   └── head.py            # 检测头
├── runs/                   # 训练结果文件
├── utils/                  # 相关工具代码
│   ├── dataset.py         # 数据集处理
│   └── utils.py           # 工具函数
├── VOCdevkit/              # 训练所用数据集
├── weights/                # 预训练权重文件夹
├── Dataset_partitioning.py # 数据转换与划分脚本
├── detect.py               # 模型检测脚本
├── requirements.txt        # 依赖库版本文件
├── train.py                # 模型训练脚本
├── val.py                  # 模型测试脚本
└── README.md               # 项目说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU加速)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 创建示例数据集

```bash
python Dataset_partitioning.py --create-sample --output VOCdevkit --sample-size 100
```

### 2. 训练模型

```bash
python train.py --config data/infrared_config.yaml --epochs 50 --batch-size 8
```

### 3. 验证模型

```bash
python val.py --model runs/train/best_model.pth --data VOCdevkit/val.txt
```

### 4. 检测目标

```bash
python detect.py --model runs/train/best_model.pth --source images/ --output inference/
```

## 详细使用说明

### 数据准备

项目支持多种数据格式：
- **VOC格式**: XML标注文件
- **COCO格式**: JSON标注文件
- **YOLO格式**: TXT标注文件

使用数据转换脚本：
```bash
# 转换现有数据集
python Dataset_partitioning.py --input /path/to/dataset --output VOCdevkit

# 创建示例数据集
python Dataset_partitioning.py --create-sample --output VOCdevkit
```

### 模型训练

训练参数说明：
- `--config`: 配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备选择（auto/cpu/cuda）
- `--save-dir`: 保存目录

训练示例：
```bash
python train.py \
    --config data/infrared_config.yaml \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --device auto \
    --save-dir runs/train
```

### 模型验证

验证参数说明：
- `--model`: 模型权重文件路径
- `--data`: 验证数据路径
- `--conf-thres`: 置信度阈值
- `--iou-thres`: NMS IoU阈值

验证示例：
```bash
python val.py \
    --model runs/train/best_model.pth \
    --data VOCdevkit/val.txt \
    --conf-thres 0.25 \
    --iou-thres 0.45
```

### 目标检测

检测参数说明：
- `--model`: 模型权重文件路径
- `--source`: 输入图像路径或目录
- `--output`: 输出目录
- `--conf-thres`: 置信度阈值
- `--iou-thres`: NMS IoU阈值

检测示例：
```bash
python detect.py \
    --model runs/train/best_model.pth \
    --source images/ \
    --output inference/ \
    --conf-thres 0.25 \
    --iou-thres 0.45
```

## 模型架构

### 骨干网络
- **ResNet风格**: 使用残差连接的特征提取网络
- **多尺度特征**: 提取不同层级的特征信息
- **全局特征**: 使用全局平均池化获取全局上下文

### Transformer检测头
- **多头注意力**: 捕获长距离依赖关系
- **位置编码**: 为序列添加位置信息
- **特征融合**: 结合全局和局部特征

### YOLO检测头
- **多尺度检测**: 在不同特征图上进行检测
- **Anchor机制**: 使用预定义的锚框
- **实时检测**: 快速的目标检测能力

## 配置说明

主要配置参数：

```yaml
# 数据集配置
dataset:
  nc: 1                    # 类别数量
  names: ['infrared_target'] # 类别名称

# 模型配置
model:
  backbone: 'resnet50'     # 骨干网络
  neck: 'fpn'             # 特征金字塔网络
  head: 'transformer'     # 检测头类型
  input_size: [512, 512]  # 输入图像尺寸

# 训练配置
train:
  batch_size: 8           # 批次大小
  epochs: 100             # 训练轮数
  learning_rate: 0.001    # 学习率
  weight_decay: 0.0005    # 权重衰减

# 检测配置
detection:
  conf_thres: 0.25        # 置信度阈值
  iou_thres: 0.45         # NMS IoU阈值
  max_det: 100            # 最大检测数量
```

## 性能指标

模型评估指标：
- **mAP**: 平均精度
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数
- **推理时间**: 单张图像处理时间

## 创新点

1. **混合架构设计**: 结合CNN和Transformer的优势
2. **红外图像优化**: 针对红外图像特点的数据增强策略
3. **多尺度特征融合**: 使用FPN进行特征金字塔融合
4. **注意力机制**: 使用Transformer捕获全局上下文信息
5. **弱小目标检测**: 专门针对弱小目标进行优化

## 常见问题

### Q: 如何调整检测阈值？
A: 使用 `--conf-thres` 和 `--iou-thres` 参数调整检测敏感度。

### Q: 如何提高检测精度？
A: 可以增加训练数据、调整模型参数、使用预训练权重等。

### Q: 支持哪些图像格式？
A: 支持JPG、PNG、BMP、TIFF等常见格式。

### Q: 如何自定义类别？
A: 修改配置文件中的 `dataset.names` 和 `dataset.nc` 参数。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请提交Issue或联系项目维护者。