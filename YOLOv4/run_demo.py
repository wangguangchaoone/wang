#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外弱小目标检测演示脚本
展示完整的工作流程
"""

import os
import sys
import argparse
import subprocess
import time

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"✅ {description} 完成！耗时: {end_time - start_time:.2f}秒")
        if result.stdout:
            print("输出:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败！")
        print(f"错误代码: {e.returncode}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return False

def check_dependencies():
    """检查依赖"""
    print("检查项目依赖...")
    
    required_files = [
        'requirements.txt',
        'data/infrared_config.yaml',
        'models/backbone.py',
        'models/head.py',
        'utils/dataset.py',
        'utils/utils.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少以下文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 项目文件检查通过")
    return True

def create_sample_dataset():
    """创建示例数据集"""
    command = "python Dataset_partitioning.py --create-sample --output VOCdevkit --sample-size 50"
    return run_command(command, "创建示例数据集")

def test_model():
    """测试模型"""
    command = "python test_model.py"
    return run_command(command, "测试模型架构")

def train_model():
    """训练模型"""
    command = "python train.py --config data/infrared_config.yaml --epochs 5 --batch-size 4 --save-dir runs/demo"
    return run_command(command, "训练模型（5个epoch）")

def validate_model():
    """验证模型"""
    if not os.path.exists("runs/demo/best_model.pth"):
        print("❌ 训练模型文件不存在，跳过验证")
        return False
    
    command = "python val.py --model runs/demo/best_model.pth --data VOCdevkit/val.txt --output runs/demo/val"
    return run_command(command, "验证模型性能")

def detect_targets():
    """检测目标"""
    if not os.path.exists("runs/demo/best_model.pth"):
        print("❌ 训练模型文件不存在，跳过检测")
        return False
    
    # 创建测试图像目录
    os.makedirs("images", exist_ok=True)
    
    # 如果没有测试图像，复制一些示例图像
    if not os.listdir("images"):
        print("创建测试图像...")
        import numpy as np
        import cv2
        
        for i in range(5):
            # 创建随机红外图像
            img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            
            # 添加一些目标
            for _ in range(np.random.randint(1, 4)):
                x = np.random.randint(50, 462)
                y = np.random.randint(50, 462)
                size = np.random.randint(10, 30)
                cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), 255, -1)
            
            cv2.imwrite(f"images/test_{i:02d}.jpg", img)
        
        print("✅ 测试图像创建完成")
    
    command = "python detect.py --model runs/demo/best_model.pth --source images/ --output runs/demo/inference"
    return run_command(command, "执行目标检测")

def show_results():
    """显示结果"""
    print(f"\n{'='*60}")
    print("演示完成！结果总结")
    print(f"{'='*60}")
    
    # 检查生成的文件
    results = []
    
    if os.path.exists("VOCdevkit"):
        train_count = len(os.listdir("VOCdevkit/images/train")) if os.path.exists("VOCdevkit/images/train") else 0
        val_count = len(os.listdir("VOCdevkit/images/val")) if os.path.exists("VOCdevkit/images/val") else 0
        results.append(f"数据集: 训练集 {train_count} 张，验证集 {val_count} 张")
    
    if os.path.exists("runs/demo/best_model.pth"):
        model_size = os.path.getsize("runs/demo/best_model.pth") / (1024 * 1024)
        results.append(f"训练模型: {model_size:.2f} MB")
    
    if os.path.exists("runs/demo/inference"):
        inference_count = len(os.listdir("runs/demo/inference"))
        results.append(f"检测结果: {inference_count} 张图像")
    
    for result in results:
        print(f"✅ {result}")
    
    print(f"\n下一步操作:")
    print("1. 查看检测结果: ls runs/demo/inference/")
    print("2. 使用自己的数据: python Dataset_partitioning.py --input /path/to/your/data --output VOCdevkit")
    print("3. 训练更多轮数: python train.py --config data/infrared_config.yaml --epochs 100")
    print("4. 调整检测参数: python detect.py --model runs/demo/best_model.pth --source your_images/ --conf-thres 0.3")

def main():
    parser = argparse.ArgumentParser(description='红外弱小目标检测演示')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip-detect', action='store_true', help='跳过检测步骤')
    parser.add_argument('--full-demo', action='store_true', help='完整演示（包括训练）')
    
    args = parser.parse_args()
    
    print("🚀 红外弱小目标检测项目演示")
    print("本脚本将展示完整的工作流程")
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请确保项目文件完整")
        return
    
    # 创建示例数据集
    if not create_sample_dataset():
        print("❌ 数据集创建失败")
        return
    
    # 测试模型
    if not test_model():
        print("❌ 模型测试失败")
        return
    
    # 训练模型（可选）
    if args.full_demo and not args.skip_train:
        if not train_model():
            print("❌ 模型训练失败")
            return
        
        # 验证模型
        if not validate_model():
            print("⚠️ 模型验证失败，但继续执行")
    
    # 检测目标（可选）
    if not args.skip_detect:
        if not detect_targets():
            print("❌ 目标检测失败")
            return
    
    # 显示结果
    show_results()

if __name__ == '__main__':
    main()