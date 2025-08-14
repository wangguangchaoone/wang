# YOLO-MHAF: 基于改进YOLO的花生叶片病害检测

## 项目简介

本项目实现了一种基于改进YOLOv8n模型的多头自适应融合检测算法YOLO-MHAF，用于花生叶片病害检测。

## 主要特性

- **混合轻量化策略**: 浅层网络引入Ghost模块，深层网络使用SPD-Conv模块
- **混合注意力机制**: 结合坐标注意力(CA)与高效通道注意力(ECA)
- **自适应多路径特征融合**: 融合BiFPN和CARAFE技术的AMFN网络

## 项目结构

```
├── models/
│   ├── __init__.py
│   ├── yolo_mhaf.py          # 主模型文件
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── ghost.py          # Ghost模块
│   │   ├── spd_conv.py       # SPD-Conv模块
│   │   ├── attention.py      # 注意力机制
│   │   ├── c2f_hybrid.py     # C2f-Hybrid结构
│   │   └── amfn.py          # 自适应多路径特征融合网络
│   └── losses.py             # 损失函数
├── utils/
│   ├── __init__.py
│   ├── datasets.py           # 数据处理
│   ├── metrics.py            # 评估指标
│   └── visualization.py     # 可视化工具
├── configs/
│   └── yolo_mhaf.yaml       # 模型配置
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── evaluate.py               # 评估脚本
├── requirements.txt          # 依赖包
└── README.md                # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型
```bash
python train.py --config configs/yolo_mhaf.yaml --data /path/to/dataset
```

### 推理检测
```bash
python inference.py --model weights/best.pt --source /path/to/images
```

### 评估模型
```bash
python evaluate.py --model weights/best.pt --data /path/to/test_data
```

## 性能指标

- 准确率: 93.8%
- 召回率: 92.4% 
- mAP@0.5: 96.7%
- mAP@0.5:0.95: 86.5%
- 参数量减少: 4.1×10⁵
- 内存占用减少: 0.82 MB

## 引用

如果您使用了本项目，请引用相关论文。
