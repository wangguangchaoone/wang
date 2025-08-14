"""
可视化工具
用于绘制检测结果、训练曲线等
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import seaborn as sns
from pathlib import Path
import torch


def draw_boxes(image, boxes, scores, classes, class_names, conf_threshold=0.25, line_thickness=2):
    """在图像上绘制检测框
    
    Args:
        image: 输入图像 (H, W, 3)
        boxes: 边界框 (N, 4) [x1, y1, x2, y2]
        scores: 置信度 (N,)
        classes: 类别 (N,)
        class_names: 类别名称列表
        conf_threshold: 置信度阈值
        line_thickness: 线条粗细
        
    Returns:
        image: 绘制了检测框的图像
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if len(image.shape) == 3 and image.shape[0] == 3:
        # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    # 确保是uint8格式
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 复制图像避免修改原图
    image = image.copy()
    h, w = image.shape[:2]
    
    # 定义颜色
    colors = [
        (255, 0, 0),    # 红色 - Healthy
        (0, 255, 0),    # 绿色 - Early_Blight
        (0, 0, 255),    # 蓝色 - Late_Blight
        (255, 255, 0),  # 黄色 - Leaf_Spot
        (255, 0, 255),  # 紫色 - Rust
    ]
    
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if score < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 选择颜色
        color = colors[int(cls) % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # 准备标签文本
        if int(cls) < len(class_names):
            label = f'{class_names[int(cls)]}: {score:.2f}'
        else:
            label = f'Class {int(cls)}: {score:.2f}'
        
        # 计算文本大小
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return image


def plot_results(results_dict, save_path=None, show=True):
    """绘制训练结果曲线
    
    Args:
        results_dict: 包含训练指标的字典
        save_path: 保存路径
        show: 是否显示图像
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO-MHAF Training Results', fontsize=16)
    
    # 损失曲线
    if 'train_loss' in results_dict:
        axes[0, 0].plot(results_dict['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in results_dict:
            axes[0, 0].plot(results_dict['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # mAP曲线
    if 'map50' in results_dict:
        axes[0, 1].plot(results_dict['map50'], label='mAP@0.5', color='green')
        if 'map' in results_dict:
            axes[0, 1].plot(results_dict['map'], label='mAP@0.5:0.95', color='orange')
        axes[0, 1].set_title('mAP Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 精确率和召回率
    if 'precision' in results_dict and 'recall' in results_dict:
        axes[1, 0].plot(results_dict['precision'], label='Precision', color='purple')
        axes[1, 0].plot(results_dict['recall'], label='Recall', color='brown')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # F1分数
    if 'f1' in results_dict:
        axes[1, 1].plot(results_dict['f1'], label='F1 Score', color='red')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, show=True):
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
        show: 是否显示
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_class_distribution(class_counts, class_names, save_path=None, show=True):
    """绘制类别分布图
    
    Args:
        class_counts: 每个类别的数量
        class_names: 类别名称
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(12, 6))
    
    # 条形图
    bars = plt.bar(class_names, class_counts, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    
    # 添加数值标签
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Disease Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_detection_examples(images, predictions, class_names, save_path=None, show=True, max_images=6):
    """绘制检测示例
    
    Args:
        images: 图像列表
        predictions: 预测结果列表
        class_names: 类别名称
        save_path: 保存路径
        show: 是否显示
        max_images: 最大显示图像数
    """
    n_images = min(len(images), max_images)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        # 绘制检测结果
        img_with_boxes = draw_boxes(
            images[i],
            predictions[i]['boxes'],
            predictions[i]['scores'],
            predictions[i]['classes'],
            class_names
        )
        
        axes[row, col].imshow(img_with_boxes)
        axes[row, col].set_title(f'Detection Result {i+1}')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detection examples saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_model_comparison(models_results, metric='map50', save_path=None, show=True):
    """绘制模型比较图
    
    Args:
        models_results: 模型结果字典 {model_name: results}
        metric: 比较的指标
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, results in models_results.items():
        if metric in results:
            plt.plot(results[metric], label=model_name, linewidth=2)
    
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_detection_video(image_paths, predictions, class_names, output_path, fps=5):
    """创建检测结果视频
    
    Args:
        image_paths: 图像路径列表
        predictions: 预测结果列表
        class_names: 类别名称
        output_path: 输出视频路径
        fps: 帧率
    """
    if len(image_paths) == 0:
        return
    
    # 读取第一张图像获取尺寸
    first_img = cv2.imread(image_paths[0])
    h, w, c = first_img.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path, pred in zip(image_paths, predictions):
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制检测框
        image_with_boxes = draw_boxes(
            image,
            pred['boxes'],
            pred['scores'],
            pred['classes'],
            class_names
        )
        
        # 转换为BGR格式
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
        
        # 写入视频
        out.write(image_with_boxes)
    
    out.release()
    print(f"Detection video saved to {output_path}")


def save_detection_results_html(images, predictions, class_names, save_path):
    """保存检测结果为HTML文件
    
    Args:
        images: 图像列表
        predictions: 预测结果列表
        class_names: 类别名称
        save_path: 保存路径
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO-MHAF Detection Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .result { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
            .image { max-width: 500px; }
            .detection { margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>YOLO-MHAF 花生叶片病害检测结果</h1>
    """
    
    for i, (image, pred) in enumerate(zip(images, predictions)):
        html_content += f"""
        <div class="result">
            <h3>图像 {i+1}</h3>
            <img src="data:image/png;base64,{image_to_base64(image)}" class="image">
            <h4>检测结果:</h4>
        """
        
        for box, score, cls in zip(pred['boxes'], pred['scores'], pred['classes']):
            class_name = class_names[int(cls)] if int(cls) < len(class_names) else f'Class {int(cls)}'
            html_content += f"""
            <div class="detection">
                {class_name}: {score:.3f} - Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]
            </div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML results saved to {save_path}")


def image_to_base64(image):
    """将图像转换为base64编码"""
    import base64
    from io import BytesIO
    from PIL import Image
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


if __name__ == '__main__':
    # 测试可视化功能
    import torch
    
    # 创建测试数据
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    scores = np.array([0.8, 0.6])
    classes = np.array([0, 1])
    class_names = ['Healthy', 'Early_Blight', 'Late_Blight', 'Leaf_Spot', 'Rust']
    
    # 测试绘制检测框
    result_image = draw_boxes(image, boxes, scores, classes, class_names)
    
    print("Visualization test completed!")
    print(f"Original image shape: {image.shape}")
    print(f"Result image shape: {result_image.shape}")