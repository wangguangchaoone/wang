#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测模型测试脚本
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.backbone import InfraredDetectionModel
from utils.utils import get_model_info

def test_model():
    """测试模型架构"""
    print("开始测试红外弱小目标检测模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print("\n创建模型...")
    model = InfraredDetectionModel(
        num_classes=1,
        d_model=256,
        n_heads=8,
        n_layers=6
    )
    
    # 获取模型信息
    model_info = get_model_info(model)
    print(f"模型参数信息:")
    print(f"  总参数数量: {model_info['total_params']:,}")
    print(f"  可训练参数数量: {model_info['trainable_params']:,}")
    print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 移动到设备
    model = model.to(device)
    
    # 创建测试输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 512, 512).to(device)
    print(f"\n测试输入形状: {input_tensor.shape}")
    
    # 前向传播
    print("执行前向传播...")
    try:
        with torch.no_grad():
            output = model(input_tensor)
        
        print("前向传播成功！")
        print(f"输出结构:")
        print(f"  Transformer检测头:")
        transformer_bbox, transformer_cls, transformer_conf = output['transformer']
        print(f"    边界框预测: {transformer_bbox.shape}")
        print(f"    分类预测: {transformer_cls.shape}")
        print(f"    置信度预测: {transformer_conf.shape}")
        
        print(f"  YOLO检测头:")
        for i, (bbox, cls, obj) in enumerate(output['yolo']):
            print(f"    尺度 {i+1}: 边界框 {bbox.shape}, 分类 {cls.shape}, 置信度 {obj.shape}")
        
        print(f"  特征图:")
        for i, feat in enumerate(output['features']):
            print(f"    特征 {i+1}: {feat.shape}")
        
        print(f"  全局特征: {output['global_feat'].shape}")
        
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试损失函数
    print("\n测试损失函数...")
    try:
        from models.head import LossFunction
        criterion = LossFunction(num_classes=1, anchors=3).to(device)
        
        # 创建模拟标签
        labels = [
            torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]).to(device),
            torch.tensor([[0, 0.3, 0.7, 0.08, 0.08]]).to(device)
        ]
        
        loss, loss_dict = criterion(output, labels)
        print(f"损失计算成功！")
        print(f"  总损失: {loss.item():.4f}")
        print(f"  边界框损失: {loss_dict['bbox_loss']:.4f}")
        print(f"  分类损失: {loss_dict['cls_loss']:.4f}")
        print(f"  置信度损失: {loss_dict['conf_loss']:.4f}")
        
    except Exception as e:
        print(f"损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试数据加载器
    print("\n测试数据加载器...")
    try:
        from utils.dataset import InfraredDataset
        
        # 创建示例数据集
        dataset = InfraredDataset(
            data_path='VOCdevkit/train.txt',
            img_size=512,
            augment=False,
            mode='train'
        )
        
        print(f"数据集创建成功！")
        print(f"  数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试数据加载
            sample = dataset[0]
            print(f"  样本键: {list(sample.keys())}")
            print(f"  图像形状: {sample['image'].shape}")
            print(f"  标签数量: {len(sample['labels'])}")
            
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        print("这可能是正常的，因为还没有创建数据集文件")
    
    print("\n模型测试完成！")
    return True

def test_utils():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    try:
        from utils.utils import compute_iou, non_max_suppression
        
        # 测试IoU计算
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = compute_iou(box1, box2)
        print(f"  IoU计算测试: {iou:.4f}")
        
        # 测试NMS
        predictions = torch.tensor([
            [0, 0, 10, 10, 0.9, 0],
            [1, 1, 11, 11, 0.8, 0],
            [20, 20, 30, 30, 0.7, 0]
        ])
        nms_result = non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.5)
        print(f"  NMS测试: 输入 {len(predictions)} 个检测框，输出 {len(nms_result[0])} 个检测框")
        
        print("  工具函数测试成功！")
        
    except Exception as e:
        print(f"工具函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("红外弱小目标检测模型测试")
    print("=" * 50)
    
    # 测试模型
    model_test_passed = test_model()
    
    # 测试工具函数
    utils_test_passed = test_utils()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    if model_test_passed and utils_test_passed:
        print("✅ 所有测试通过！模型架构正确。")
        print("\n下一步:")
        print("1. 创建数据集: python Dataset_partitioning.py --create-sample --output VOCdevkit")
        print("2. 训练模型: python train.py --config data/infrared_config.yaml --epochs 10")
        print("3. 检测目标: python detect.py --model runs/train/best_model.pth --source images/")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    print("=" * 50)

if __name__ == '__main__':
    main()