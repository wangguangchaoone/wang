"""
YOLO-MHAF模型测试脚本
验证模型各个组件是否正常工作
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo_mhaf_simple import create_yolo_mhaf_n, create_yolo_mhaf_s, create_yolo_mhaf_m
from models.modules.ghost import GhostConv, C2f_Ghost
from models.modules.spd_conv import SPDConv, C2f_SPD
from models.modules.attention import HybridAttention, CoordAtt, ECA
from models.modules.c2f_hybrid import C2f_Hybrid, SPPF_Hybrid
from models.modules.amfn import AMFN_Lite, CARAFE, BiFPN
from models.losses import YOLOLoss
from utils.metrics import bbox_iou, compute_metrics
from utils.visualization import draw_boxes


def test_ghost_module():
    """测试Ghost模块"""
    print("Testing Ghost modules...")
    
    # 测试GhostConv
    ghost_conv = GhostConv(64, 128, k=3, s=1)
    x = torch.randn(2, 64, 32, 32)
    y = ghost_conv(x)
    print(f"GhostConv: {x.shape} -> {y.shape}")
    assert y.shape == (2, 128, 32, 32), f"Expected (2, 128, 32, 32), got {y.shape}"
    
    # 测试C2f_Ghost
    c2f_ghost = C2f_Ghost(128, 256, n=2)
    x = torch.randn(2, 128, 16, 16)
    y = c2f_ghost(x)
    print(f"C2f_Ghost: {x.shape} -> {y.shape}")
    assert y.shape == (2, 256, 16, 16), f"Expected (2, 256, 16, 16), got {y.shape}"
    
    print("✓ Ghost modules test passed!")


def test_spd_conv():
    """测试SPD-Conv模块"""
    print("Testing SPD-Conv modules...")
    
    # 测试SPDConv
    spd_conv = SPDConv(64, 128, k=3, s=2)
    x = torch.randn(2, 64, 32, 32)
    y = spd_conv(x)
    print(f"SPDConv: {x.shape} -> {y.shape}")
    # SPD-Conv with stride=2 should reduce spatial dimensions by half
    expected_shape = (2, 128, 16, 16)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    # 测试C2f_SPD
    c2f_spd = C2f_SPD(128, 256, n=2)
    x = torch.randn(2, 128, 16, 16)
    y = c2f_spd(x)
    print(f"C2f_SPD: {x.shape} -> {y.shape}")
    assert y.shape == (2, 256, 16, 16), f"Expected (2, 256, 16, 16), got {y.shape}"
    
    print("✓ SPD-Conv modules test passed!")


def test_attention_modules():
    """测试注意力模块"""
    print("Testing attention modules...")
    
    x = torch.randn(2, 256, 16, 16)
    
    # 测试CoordAtt
    ca = CoordAtt(256, 256)
    y_ca = ca(x)
    print(f"CoordAtt: {x.shape} -> {y_ca.shape}")
    assert y_ca.shape == x.shape, f"Expected {x.shape}, got {y_ca.shape}"
    
    # 测试ECA
    eca = ECA(256)
    y_eca = eca(x)
    print(f"ECA: {x.shape} -> {y_eca.shape}")
    assert y_eca.shape == x.shape, f"Expected {x.shape}, got {y_eca.shape}"
    
    # 测试HybridAttention
    hybrid_att = HybridAttention(256)
    y_hybrid = hybrid_att(x)
    print(f"HybridAttention: {x.shape} -> {y_hybrid.shape}")
    assert y_hybrid.shape == x.shape, f"Expected {x.shape}, got {y_hybrid.shape}"
    
    print("✓ Attention modules test passed!")


def test_c2f_hybrid():
    """测试C2f-Hybrid模块"""
    print("Testing C2f-Hybrid modules...")
    
    # 测试C2f_Hybrid
    c2f_hybrid = C2f_Hybrid(256, 512, n=3, use_attention=True, attention_type='hybrid')
    x = torch.randn(2, 256, 16, 16)
    y = c2f_hybrid(x)
    print(f"C2f_Hybrid: {x.shape} -> {y.shape}")
    assert y.shape == (2, 512, 16, 16), f"Expected (2, 512, 16, 16), got {y.shape}"
    
    # 测试SPPF_Hybrid
    sppf_hybrid = SPPF_Hybrid(512, 512, use_attention=True)
    x = torch.randn(2, 512, 16, 16)
    y = sppf_hybrid(x)
    print(f"SPPF_Hybrid: {x.shape} -> {y.shape}")
    assert y.shape == (2, 512, 16, 16), f"Expected (2, 512, 16, 16), got {y.shape}"
    
    print("✓ C2f-Hybrid modules test passed!")


def test_amfn():
    """测试AMFN模块"""
    print("Testing AMFN modules...")
    
    # 测试CARAFE
    carafe = CARAFE(256, scale=2)
    x = torch.randn(2, 256, 8, 8)
    y = carafe(x)
    print(f"CARAFE: {x.shape} -> {y.shape}")
    expected_shape = (2, 256, 16, 16)  # 上采样2倍
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    # 测试AMFN_Lite
    amfn = AMFN_Lite([256, 512, 1024], 256, num_levels=3)
    inputs = [
        torch.randn(2, 256, 32, 32),   # P3
        torch.randn(2, 512, 16, 16),   # P4
        torch.randn(2, 1024, 8, 8),    # P5
    ]
    outputs = amfn(inputs)
    print(f"AMFN_Lite: {len(inputs)} inputs -> {len(outputs)} outputs")
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
    
    print("✓ AMFN modules test passed!")


def test_complete_model():
    """测试完整模型"""
    print("Testing complete YOLO-MHAF models...")
    
    # 测试不同规模的模型
    models = [
        ('yolo_mhaf_n', create_yolo_mhaf_n),
        ('yolo_mhaf_s', create_yolo_mhaf_s),
        ('yolo_mhaf_m', create_yolo_mhaf_m),
    ]
    
    x = torch.randn(1, 3, 640, 640)
    
    for model_name, model_fn in models:
        print(f"\nTesting {model_name}...")
        model = model_fn(nc=5)
        model.eval()
        
        with torch.no_grad():
            outputs = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Number of output layers: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
        
        # 获取模型信息
        info = model.get_model_info()
        print(f"Parameters: {info['total_params']:,}")
        print(f"Model size: {info['model_size_mb']:.2f} MB")
        
        print(f"✓ {model_name} test passed!")


def test_loss_function():
    """测试损失函数"""
    print("Testing YOLO loss function...")
    
    # 创建测试数据
    batch_size = 2
    nc = 5
    
    # 模拟预测结果
    predictions = [
        torch.randn(batch_size, 1, 20, 20, nc + 5),
        torch.randn(batch_size, 1, 40, 40, nc + 5),
        torch.randn(batch_size, 1, 80, 80, nc + 5),
    ]
    
    # 模拟目标
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],  # image_id, class, x, y, w, h
        [1, 1, 0.3, 0.4, 0.1, 0.15],
    ])
    
    # 计算损失
    criterion = YOLOLoss(nc=nc)
    loss, loss_items = criterion(predictions, targets)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_items}")
    
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    
    print("✓ Loss function test passed!")


def test_metrics():
    """测试评估指标"""
    print("Testing evaluation metrics...")
    
    # 测试IoU计算
    box1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]])
    box2 = torch.tensor([[15, 15, 55, 55], [10, 10, 30, 30]])
    
    iou = bbox_iou(box1, box2)
    print(f"IoU matrix:\n{iou}")
    
    assert iou.shape == (2, 2), f"Expected (2, 2), got {iou.shape}"
    assert torch.all(iou >= 0) and torch.all(iou <= 1), "IoU values should be in [0, 1]"
    
    print("✓ Metrics test passed!")


def test_visualization():
    """测试可视化功能"""
    print("Testing visualization functions...")
    
    # 创建测试图像和检测结果
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    scores = np.array([0.8, 0.6])
    classes = np.array([0, 1])
    class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Spot', 'Rust']
    
    # 测试绘制检测框
    result_image = draw_boxes(image, boxes, scores, classes, class_names)
    
    print(f"Original image shape: {image.shape}")
    print(f"Result image shape: {result_image.shape}")
    
    assert result_image.shape == image.shape, f"Expected {image.shape}, got {result_image.shape}"
    assert result_image.dtype == np.uint8, f"Expected uint8, got {result_image.dtype}"
    
    print("✓ Visualization test passed!")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("YOLO-MHAF Model Component Tests")
    print("=" * 60)
    
    try:
        test_ghost_module()
        print()
        
        test_spd_conv()
        print()
        
        test_attention_modules()
        print()
        
        test_c2f_hybrid()
        print()
        
        test_amfn()
        print()
        
        test_complete_model()
        print()
        
        test_loss_function()
        print()
        
        test_metrics()
        print()
        
        test_visualization()
        print()
        
        print("=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
        print("\nYOLO-MHAF implementation is ready for training!")
        print("Next steps:")
        print("1. Prepare your peanut disease dataset")
        print("2. Run training: python train.py --data-path /path/to/dataset")
        print("3. Run inference: python inference.py --model weights/best.pt --source /path/to/images")
        print("4. Evaluate model: python evaluate.py --model-path weights/best.pt --data-path /path/to/dataset")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)