"""
评估指标计算
包含mAP、精确率、召回率等目标检测评估指标
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from collections import defaultdict


def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    """计算两个边界框的IoU
    
    Args:
        box1: (N, 4) 或 (4,)
        box2: (M, 4) 或 (4,)
        x1y1x2y2: 如果True，box格式为(x1, y1, x2, y2)；否则为(x, y, w, h)
        
    Returns:
        iou: (N, M) 或 (N,) 或 scalar
    """
    # 获取box的角点坐标
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T if box1.ndim > 1 else box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T if box2.ndim > 1 else box2
    else:
        # 转换(x, y, w, h)到(x1, y1, x2, y2)
        b1_x1, b1_y1 = (box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2) if box1.ndim > 1 else (box1[0] - box1[2] / 2, box1[1] - box1[3] / 2)
        b1_x2, b1_y2 = (box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2) if box1.ndim > 1 else (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
        b2_x1, b2_y1 = (box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2) if box2.ndim > 1 else (box2[0] - box2[2] / 2, box2[1] - box2[3] / 2)
        b2_x2, b2_y2 = (box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2) if box2.ndim > 1 else (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)

    # 交集区域
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 并集区域
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou


def compute_ap(recall, precision):
    """计算平均精度(AP)
    
    Args:
        recall: 召回率数组
        precision: 精确率数组
        
    Returns:
        ap: 平均精度
    """
    # 添加边界点
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 计算精度包络
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 查找召回率变化点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算曲线下面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """计算每个类别的AP
    
    Args:
        tp: 真正例数组
        conf: 置信度数组
        pred_cls: 预测类别数组
        target_cls: 目标类别数组
        
    Returns:
        ap: 每个类别的AP
        p: 精确率
        r: 召回率
    """
    # 按置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 查找唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    # 创建精度和召回率曲线
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 标签数量
        n_p = i.sum()  # 预测数量

        if n_p == 0 or n_l == 0:
            continue

        # 累积FP和TP
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 召回率
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # 精确率
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # 计算AP
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # 计算F1分数
    f1 = 2 * p * r / (p + r + eps)

    return ap, f1, p, r


def process_batch(detections, labels, iouv):
    """处理一个批次的检测结果
    
    Args:
        detections: 检测结果 (N, 6) [x1, y1, x2, y2, conf, class]
        labels: 标签 (M, 5) [class, x1, y1, x2, y2]
        iouv: IoU阈值向量
        
    Returns:
        correct: 正确检测的布尔矩阵 (N, 10)
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return correct
    
    # 计算IoU
    iou = bbox_iou(labels[:, 1:], detections[:, :4])
    
    # 匹配检测和标签
    correct_class = labels[:, 0:1] == detections[:, 5]
    
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            
    return torch.tensor(correct, dtype=torch.bool, device=detections.device)


def compute_metrics(predictions, targets, nc, iou_thres=0.5):
    """计算检测指标
    
    Args:
        predictions: 预测结果列表
        targets: 目标列表
        nc: 类别数量
        iou_thres: IoU阈值
        
    Returns:
        metrics: 包含各种指标的字典
    """
    # IoU阈值向量
    iouv = torch.linspace(0.5, 0.95, 10)  # mAP@0.5:0.95的IoU向量
    niou = iouv.numel()
    
    # 初始化统计
    stats = []
    
    for pred, target in zip(predictions, targets):
        if isinstance(pred, list) and len(pred) > 0:
            # 如果预测是列表格式，取第一个
            pred = pred[0] if len(pred) > 0 else torch.empty((0, 6))
        
        if isinstance(target, torch.Tensor) and target.numel() > 0:
            # 转换目标格式
            target = target.cpu().numpy()
            if target.shape[1] == 6:  # [batch_idx, class, x, y, w, h]
                # 转换为 [class, x1, y1, x2, y2]
                target_boxes = target[:, 2:]  # x, y, w, h
                target_boxes[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2  # x1
                target_boxes[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2  # y1
                target_boxes[:, 2] = target_boxes[:, 0] + target_boxes[:, 2]      # x2
                target_boxes[:, 3] = target_boxes[:, 1] + target_boxes[:, 3]      # y2
                target = np.column_stack([target[:, 1], target_boxes])
        
        if isinstance(pred, torch.Tensor) and pred.numel() > 0:
            pred = pred.cpu().numpy()
            
        # 处理空预测
        if not isinstance(pred, np.ndarray) or pred.shape[0] == 0:
            if isinstance(target, np.ndarray) and target.shape[0] > 0:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), 
                             torch.Tensor(), torch.Tensor(), target[:, 0]))
            continue
            
        # 处理空目标
        if not isinstance(target, np.ndarray) or target.shape[0] == 0:
            stats.append((torch.zeros(pred.shape[0], niou, dtype=torch.bool), 
                         torch.tensor(pred[:, 4]), torch.tensor(pred[:, 5]), torch.Tensor()))
            continue
            
        # 处理批次
        correct = process_batch(torch.tensor(pred), torch.tensor(target), iouv)
        stats.append((correct, torch.tensor(pred[:, 4]), torch.tensor(pred[:, 5]), torch.tensor(target[:, 0])))
    
    # 合并统计
    if len(stats) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map': 0.0,
            'f1': 0.0
        }
    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map_score = p.mean(), r.mean(), ap50.mean(), ap.mean()
        
        return {
            'precision': float(mp),
            'recall': float(mr),
            'map50': float(map50),
            'map': float(map_score),
            'f1': float(f1.mean()),
            'ap_per_class': ap.tolist(),
            'ap50_per_class': ap50.tolist()
        }
    else:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map': 0.0,
            'f1': 0.0
        }


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """计算每个类别的AP
    
    Returns:
        tp: 真正例
        fp: 假正例  
        p: 精确率
        r: 召回率
        f1: F1分数
        ap: 平均精度
        unique_classes: 唯一类别
    """
    # 按置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 查找唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    # 创建精度和召回率曲线
    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 标签数量
        n_p = i.sum()  # 预测数量

        if n_p == 0 or n_l == 0:
            continue

        # 累积FP和TP
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 召回率
        recall = tpc / (n_l + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # 精确率
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # 计算AP
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # 计算F1分数
    f1 = 2 * p * r / (p + r + eps)
    
    names = [f'Class {i}' for i in unique_classes] if len(names) == 0 else [names[int(i)] for i in unique_classes]
    
    if plot:
        plot_pr_curve(px, py, ap, save_dir, names)
        plot_mc_curve(px, f1, save_dir, names, ylabel='F1')
        plot_mc_curve(px, p, save_dir, names, ylabel='Precision')
        plot_mc_curve(px, r, save_dir, names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def smooth(y, f=0.05):
    """对数组进行平滑处理"""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, save_dir=Path('.'), names=(), on_plot=None):
    """绘制PR曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir / 'PR_curve.png', dpi=250)
    plt.close(fig)

    if on_plot:
        on_plot(save_dir / 'PR_curve.png')


def plot_mc_curve(px, py, save_dir=Path('.'), names=(), ylabel='Metric', on_plot=None):
    """绘制指标曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel.lower()}_curve.png', dpi=250)
    plt.close(fig)

    if on_plot:
        on_plot(save_dir / f'{ylabel.lower()}_curve.png')


if __name__ == '__main__':
    # 测试IoU计算
    box1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
    box2 = torch.tensor([[2, 2, 12, 12], [0, 0, 5, 5]])
    
    iou = bbox_iou(box1, box2)
    print(f"IoU matrix:\n{iou}")
    
    # 测试AP计算
    recall = np.array([0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    precision = np.array([1.0, 1.0, 0.8, 0.6, 0.6, 0.4, 0.2])
    
    ap = compute_ap(recall, precision)
    print(f"AP: {ap:.4f}")