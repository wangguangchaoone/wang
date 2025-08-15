# 快速开始指南

## 🚀 一键运行演示

```bash
# 完整演示（包括训练）
python run_demo.py --full-demo

# 仅演示检测功能（跳过训练）
python run_demo.py --skip-train

# 仅演示数据准备和模型测试
python run_demo.py --skip-train --skip-detect
```

## 📋 手动步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试模型
```bash
python test_model.py
```

### 3. 创建示例数据集
```bash
python Dataset_partitioning.py --create-sample --output VOCdevkit --sample-size 100
```

### 4. 训练模型
```bash
python train.py --config data/infrared_config.yaml --epochs 10 --batch-size 8
```

### 5. 验证模型
```bash
python val.py --model runs/train/best_model.pth --data VOCdevkit/val.txt
```

### 6. 检测目标
```bash
python detect.py --model runs/train/best_model.pth --source images/ --output inference/
```

## 🔧 自定义配置

编辑 `data/infrared_config.yaml` 文件来调整：
- 模型参数
- 训练参数
- 检测阈值
- 数据增强策略

## 📁 项目结构

```
YOLOv4/
├── models/          # 模型架构
├── utils/           # 工具函数
├── data/            # 配置文件
├── VOCdevkit/       # 数据集
├── runs/            # 训练结果
├── inference/       # 检测结果
└── weights/         # 预训练权重
```

## 💡 使用提示

1. **首次运行**: 建议先运行 `python test_model.py` 检查环境
2. **快速测试**: 使用 `--sample-size 10` 创建小数据集快速测试
3. **GPU加速**: 确保安装了CUDA版本的PyTorch
4. **内存优化**: 如果内存不足，减小 `batch-size` 参数

## 🆘 常见问题

**Q: 提示缺少依赖包？**
A: 运行 `pip install -r requirements.txt`

**Q: 训练时内存不足？**
A: 减小 `batch-size` 参数，如 `--batch-size 4`

**Q: 检测结果不理想？**
A: 调整 `--conf-thres` 和 `--iou-thres` 参数

**Q: 如何使用自己的数据？**
A: 参考 `Dataset_partitioning.py` 的用法说明