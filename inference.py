"""
YOLO-MHAF推理脚本
用于花生叶片病害检测的推理和可视化
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolo_mhaf_simple import create_yolo_mhaf_n, create_yolo_mhaf_s, create_yolo_mhaf_m
from utils.visualization import plot_results, draw_boxes


class YOLOMHAFInference:
    """YOLO-MHAF推理器"""
    
    def __init__(self, model_path, model_type='yolo_mhaf_n', device='auto', conf_thres=0.25, iou_thres=0.45):
        self.device = self.select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 花生叶片病害类别名称
        self.class_names = [
            'Healthy',           # 健康
            'Early_Blight',      # 早疫病
            'Late_Blight',       # 晚疫病
            'Leaf_Spot',         # 叶斑病
            'Rust'              # 锈病
        ]
        
        # 加载模型
        self.model = self.load_model(model_path, model_type)
        self.model.eval()
        
    def select_device(self, device):
        """选择设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def load_model(self, model_path, model_type):
        """加载模型"""
        # 创建模型
        if model_type == 'yolo_mhaf_n':
            model = create_yolo_mhaf_n(nc=len(self.class_names))
        elif model_type == 'yolo_mhaf_s':
            model = create_yolo_mhaf_s(nc=len(self.class_names))
        elif model_type == 'yolo_mhaf_m':
            model = create_yolo_mhaf_m(nc=len(self.class_names))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 加载权重
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found, using random weights")
        
        return model.to(self.device)
    
    def preprocess(self, image, img_size=640):
        """预处理图像"""
        if isinstance(image, str):
            # 从文件路径加载图像
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸
        original_shape = image.shape[:2]
        
        # 调整大小
        image = cv2.resize(image, (img_size, img_size))
        
        # 转换为tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image = image.unsqueeze(0).to(self.device)
        
        return image, original_shape
    
    def postprocess(self, outputs, original_shape, img_size=640):
        """后处理预测结果"""
        predictions = []
        
        for output in outputs:
            # output shape: [batch, anchors, grid_y, grid_x, predictions]
            batch_size, num_anchors, grid_h, grid_w, num_predictions = output.shape
            
            # 重塑为 [batch, num_boxes, predictions]
            output = output.view(batch_size, -1, num_predictions)
            
            for i in range(batch_size):
                pred = output[i]  # [num_boxes, predictions]
                
                # 提取坐标、置信度和类别概率
                boxes = pred[:, :4]  # x, y, w, h
                obj_conf = pred[:, 4]  # 目标置信度
                class_probs = pred[:, 5:]  # 类别概率
                
                # 计算类别置信度
                class_conf, class_pred = torch.max(class_probs, dim=1)
                conf = obj_conf * class_conf
                
                # 置信度过滤
                mask = conf > self.conf_thres
                boxes = boxes[mask]
                conf = conf[mask]
                class_pred = class_pred[mask]
                
                if boxes.shape[0] == 0:
                    continue
                
                # 坐标转换 (相对坐标转绝对坐标)
                boxes = self.xywh2xyxy(boxes)
                
                # 缩放到原始图像尺寸
                boxes = self.scale_coords((img_size, img_size), boxes, original_shape)
                
                # NMS
                keep = self.nms(boxes, conf, self.iou_thres)
                
                final_boxes = boxes[keep].cpu().numpy()
                final_conf = conf[keep].cpu().numpy()
                final_class = class_pred[keep].cpu().numpy()
                
                predictions.append({
                    'boxes': final_boxes,
                    'scores': final_conf,
                    'classes': final_class
                })
        
        return predictions
    
    def xywh2xyxy(self, x):
        """转换box格式从xywh到xyxy"""
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y
    
    def scale_coords(self, img1_shape, coords, img0_shape):
        """将坐标从img1_shape缩放到img0_shape"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        
        # 裁剪到图像边界
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        
        return coords
    
    def nms(self, boxes, scores, iou_threshold):
        """非极大值抑制"""
        return torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    def predict(self, image, img_size=640):
        """预测单张图像"""
        # 预处理
        processed_img, original_shape = self.preprocess(image, img_size)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(processed_img)
        
        # 后处理
        predictions = self.postprocess(outputs, original_shape, img_size)
        
        return predictions[0] if predictions else {'boxes': [], 'scores': [], 'classes': []}
    
    def predict_batch(self, images, img_size=640):
        """批量预测"""
        results = []
        for image in images:
            result = self.predict(image, img_size)
            results.append(result)
        return results
    
    def visualize(self, image, prediction, save_path=None, show=True):
        """可视化检测结果"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        boxes = prediction['boxes']
        scores = prediction['scores']
        classes = prediction['classes']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        
        for box, score, cls in zip(boxes, scores, classes):
            if len(box) == 0:
                continue
                
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # 绘制边界框
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                   edgecolor=colors[int(cls)], facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签
            label = f'{self.class_names[int(cls)]}: {score:.2f}'
            ax.text(x1, y1 - 5, label, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[int(cls)], alpha=0.7))
        
        ax.set_title('YOLO-MHAF 花生叶片病害检测结果', fontsize=16)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO-MHAF Inference')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--model-type', type=str, default='yolo_mhaf_n',
                       choices=['yolo_mhaf_n', 'yolo_mhaf_s', 'yolo_mhaf_m'],
                       help='Model type')
    parser.add_argument('--source', type=str, required=True, help='Source image or directory')
    parser.add_argument('--output', type=str, default='./runs/inference', help='Output directory')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save', action='store_true', help='Save results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化推理器
    inferencer = YOLOMHAFInference(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
    
    # 获取图像路径
    if os.path.isfile(args.source):
        image_paths = [args.source]
    elif os.path.isdir(args.source):
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(Path(args.source).glob(ext))
    else:
        raise ValueError(f"Source {args.source} not found")
    
    print(f"Found {len(image_paths)} images")
    
    # 推理
    for img_path in image_paths:
        print(f"Processing {img_path}")
        
        start_time = time.time()
        prediction = inferencer.predict(str(img_path), args.img_size)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Detected {len(prediction['boxes'])} objects")
        
        # 打印检测结果
        for i, (box, score, cls) in enumerate(zip(prediction['boxes'], 
                                                 prediction['scores'], 
                                                 prediction['classes'])):
            class_name = inferencer.class_names[int(cls)]
            print(f"  {i+1}: {class_name} ({score:.3f}) at {box}")
        
        # 可视化
        if args.show or args.save:
            save_path = os.path.join(args.output, f"result_{Path(img_path).name}") if args.save else None
            inferencer.visualize(str(img_path), prediction, save_path, args.show)


if __name__ == '__main__':
    main()