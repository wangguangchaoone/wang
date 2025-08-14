"""
YOLO-MHAF模型评估脚本
用于评估训练好的模型性能
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo_mhaf_simple import create_yolo_mhaf_n, create_yolo_mhaf_s, create_yolo_mhaf_m
from utils.datasets import create_dataloader
from utils.metrics import compute_metrics, ap_per_class
from utils.visualization import plot_confusion_matrix, plot_class_distribution
from inference import YOLOMHAFInference


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 类别名称
        self.class_names = [
            'Healthy',           # 健康
            'Early_Blight',      # 早疫病
            'Late_Blight',       # 晚疫病
            'Leaf_Spot',         # 叶斑病
            'Rust'              # 锈病
        ]
        
        # 加载模型
        self.model = self.load_model()
        
        # 创建数据加载器
        self.test_loader = create_dataloader(
            path=args.data_path,
            mode='test' if Path(args.data_path + '/images/test').exists() else 'val',
            img_size=args.img_size,
            batch_size=args.batch_size,
            workers=args.workers,
            augment=False
        )
        
    def load_model(self):
        """加载模型"""
        # 创建模型
        if self.args.model_type == 'yolo_mhaf_n':
            model = create_yolo_mhaf_n(nc=len(self.class_names))
        elif self.args.model_type == 'yolo_mhaf_s':
            model = create_yolo_mhaf_s(nc=len(self.class_names))
        elif self.args.model_type == 'yolo_mhaf_m':
            model = create_yolo_mhaf_m(nc=len(self.class_names))
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")
        
        # 加载权重
        if os.path.isfile(self.args.model_path):
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint: {self.args.model_path}")
                if 'metrics' in checkpoint:
                    print(f"Checkpoint metrics: {checkpoint['metrics']}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights: {self.args.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def evaluate(self):
        """评估模型"""
        print(f"Evaluating model on {len(self.test_loader.dataset)} samples...")
        
        all_predictions = []
        all_targets = []
        all_pred_classes = []
        all_true_classes = []
        
        # 推理器用于后处理
        inferencer = YOLOMHAFInference(
            model_path=self.args.model_path,
            model_type=self.args.model_type,
            device=self.device,
            conf_thres=self.args.conf_thres,
            iou_thres=self.args.iou_thres
        )
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(self.device, non_blocking=True).float() / 255.0
                
                # 模型推理
                outputs = self.model(images)
                
                # 批量处理每张图像
                for i in range(images.shape[0]):
                    # 获取单张图像的预测结果
                    img_outputs = [out[i:i+1] for out in outputs]
                    
                    # 后处理
                    original_shape = (self.args.img_size, self.args.img_size)
                    predictions = inferencer.postprocess(img_outputs, original_shape, self.args.img_size)
                    
                    if predictions:
                        pred = predictions[0]
                        all_predictions.append(pred)
                        
                        # 收集预测类别
                        if len(pred['classes']) > 0:
                            all_pred_classes.extend(pred['classes'])
                    else:
                        all_predictions.append({'boxes': [], 'scores': [], 'classes': []})
                    
                    # 获取目标
                    img_targets = targets[targets[:, 0] == i]  # 筛选当前图像的目标
                    if len(img_targets) > 0:
                        all_targets.append(img_targets)
                        all_true_classes.extend(img_targets[:, 1].cpu().numpy())
                    else:
                        all_targets.append(torch.empty((0, 6)))
        
        # 计算指标
        print("Computing metrics...")
        metrics = compute_metrics(all_predictions, all_targets, len(self.class_names))
        
        # 打印结果
        self.print_results(metrics)
        
        # 保存结果
        if self.args.save_results:
            self.save_results(metrics, all_pred_classes, all_true_classes)
        
        return metrics
    
    def print_results(self, metrics):
        """打印评估结果"""
        print("\n" + "="*50)
        print("YOLO-MHAF Model Evaluation Results")
        print("="*50)
        
        print(f"Model Type: {self.args.model_type}")
        print(f"Test Dataset: {len(self.test_loader.dataset)} samples")
        print(f"Image Size: {self.args.img_size}")
        print(f"Confidence Threshold: {self.args.conf_thres}")
        print(f"IoU Threshold: {self.args.iou_thres}")
        
        print("\nOverall Metrics:")
        print("-" * 30)
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1 Score:      {metrics['f1']:.4f}")
        print(f"mAP@0.5:       {metrics['map50']:.4f}")
        print(f"mAP@0.5:0.95:  {metrics['map']:.4f}")
        
        # 每个类别的结果
        if 'ap_per_class' in metrics and len(metrics['ap_per_class']) > 0:
            print("\nPer-Class AP@0.5:0.95:")
            print("-" * 30)
            for i, (class_name, ap) in enumerate(zip(self.class_names, metrics['ap_per_class'])):
                print(f"{class_name:15}: {ap:.4f}")
        
        if 'ap50_per_class' in metrics and len(metrics['ap50_per_class']) > 0:
            print("\nPer-Class AP@0.5:")
            print("-" * 30)
            for i, (class_name, ap50) in enumerate(zip(self.class_names, metrics['ap50_per_class'])):
                print(f"{class_name:15}: {ap50:.4f}")
        
        print("\n" + "="*50)
    
    def save_results(self, metrics, pred_classes, true_classes):
        """保存评估结果"""
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标到JSON
        results = {
            'model_type': self.args.model_type,
            'model_path': self.args.model_path,
            'test_samples': len(self.test_loader.dataset),
            'image_size': self.args.img_size,
            'conf_threshold': self.args.conf_thres,
            'iou_threshold': self.args.iou_thres,
            'metrics': metrics,
            'class_names': self.class_names
        }
        
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {save_dir / 'evaluation_results.json'}")
        
        # 绘制混淆矩阵（如果有足够的数据）
        if len(pred_classes) > 0 and len(true_classes) > 0:
            try:
                # 将预测和真实类别对齐（简化处理）
                min_len = min(len(pred_classes), len(true_classes))
                if min_len > 0:
                    plot_confusion_matrix(
                        true_classes[:min_len], 
                        pred_classes[:min_len], 
                        self.class_names,
                        save_path=save_dir / 'confusion_matrix.png',
                        show=False
                    )
            except Exception as e:
                print(f"Warning: Could not generate confusion matrix: {e}")
        
        # 绘制类别分布
        try:
            if len(true_classes) > 0:
                class_counts = [true_classes.count(i) for i in range(len(self.class_names))]
                plot_class_distribution(
                    class_counts, 
                    self.class_names,
                    save_path=save_dir / 'class_distribution.png',
                    show=False
                )
        except Exception as e:
            print(f"Warning: Could not generate class distribution plot: {e}")
    
    def compare_models(self, model_paths, model_types):
        """比较多个模型"""
        results = {}
        
        for model_path, model_type in zip(model_paths, model_types):
            print(f"\nEvaluating {model_type}...")
            
            # 临时修改参数
            original_path = self.args.model_path
            original_type = self.args.model_type
            
            self.args.model_path = model_path
            self.args.model_type = model_type
            
            # 重新加载模型
            self.model = self.load_model()
            
            # 评估
            metrics = self.evaluate()
            results[model_type] = metrics
            
            # 恢复参数
            self.args.model_path = original_path
            self.args.model_type = original_type
        
        # 打印比较结果
        self.print_comparison(results)
        
        return results
    
    def print_comparison(self, results):
        """打印模型比较结果"""
        print("\n" + "="*70)
        print("Model Comparison Results")
        print("="*70)
        
        # 表头
        print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'mAP@0.5':<10} {'mAP':<10}")
        print("-" * 70)
        
        # 每个模型的结果
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} {metrics['map50']:<10.4f} {metrics['map']:<10.4f}")
        
        print("="*70)
        
        # 找出最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['map'])
        print(f"Best model: {best_model[0]} (mAP: {best_model[1]['map']:.4f})")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO-MHAF Model Evaluation')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--model-type', type=str, default='yolo_mhaf_n',
                       choices=['yolo_mhaf_n', 'yolo_mhaf_s', 'yolo_mhaf_m'],
                       help='Model type')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True, help='Dataset path')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    # 推理参数
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    
    # 保存参数
    parser.add_argument('--save-results', action='store_true', help='Save evaluation results')
    parser.add_argument('--save-dir', type=str, default='./runs/eval', help='Save directory')
    
    # 比较参数
    parser.add_argument('--compare-models', nargs='+', help='Compare multiple models (paths)')
    parser.add_argument('--compare-types', nargs='+', help='Model types for comparison')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建保存目录
    if args.save_results:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = ModelEvaluator(args)
    
    if args.compare_models and args.compare_types:
        # 比较多个模型
        if len(args.compare_models) != len(args.compare_types):
            raise ValueError("Number of model paths must match number of model types")
        
        results = evaluator.compare_models(args.compare_models, args.compare_types)
        
        # 保存比较结果
        if args.save_results:
            with open(os.path.join(args.save_dir, 'model_comparison.json'), 'w') as f:
                json.dump(results, f, indent=2)
    else:
        # 单个模型评估
        metrics = evaluator.evaluate()
        
        print(f"\nEvaluation completed!")
        print(f"Final mAP@0.5: {metrics['map50']:.4f}")
        print(f"Final mAP@0.5:0.95: {metrics['map']:.4f}")


if __name__ == '__main__':
    main()