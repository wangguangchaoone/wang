"""
YOLO损失函数
包含目标检测的各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOLoss(nn.Module):
    """YOLO损失函数"""
    
    def __init__(self, nc=80, anchors=None, device=None):
        super().__init__()
        self.nc = nc  # 类别数
        self.no = nc + 5  # 每个anchor的输出数量
        self.nl = 3  # 检测层数
        self.na = 1  # anchor数量
        
        # 损失权重
        self.hyp = {
            'box': 0.05,
            'cls': 0.5,
            'obj': 1.0,
            'anchor_t': 4.0,
        }
        
        # 损失函数
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        
        # Focal loss
        self.fl_gamma = 1.5
        if self.fl_gamma > 0:
            self.BCEcls, self.BCEobj = FocalLoss(self.BCEcls, self.fl_gamma), FocalLoss(self.BCEobj, self.fl_gamma)
        
        self.device = device or torch.device('cpu')
        self.autobalance = False
        self.sort_obj_iou = False
        
    def forward(self, p, targets):
        """
        Args:
            p: 预测结果列表，每个元素shape为[batch, anchors, grid_h, grid_w, no]
            targets: 目标tensor，shape为[num_targets, 6] (image, class, x, y, w, h)
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        
        # 计算损失
        for i, pi in enumerate(p):  # 层级索引
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 目标obj
            
            n = b.shape[0]  # 目标数量
            if n:
                ps = pi[b, a, gj, gi]  # 预测子集，对应目标
                
                # 回归损失
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测框
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou损失
                
                # 目标性损失
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - 0.5) + 0.5 * score_iou  # iou比例
                
                # 分类损失
                if self.nc > 1:  # 类别损失（只有当类别数>1时）
                    t = torch.full_like(ps[:, 5:], 0, device=device)  # 目标
                    t[range(n), tcls[i]] = 1
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * (1.4 if i == 0 else 1.0)  # obj损失
            
        if self.autobalance:
            self.balance_losses(lobj, lcls, lbox)
            
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
    
    def build_targets(self, p, targets):
        """构建训练目标"""
        na, nt = self.na, targets.shape[0]  # anchor数量，目标数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # 归一化到网格空间的增益
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 添加anchor indices
        
        g = 0.5  # 偏移
        off = torch.tensor([[0, 0],
                           [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                           # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                           ], device=targets.device).float() * g  # offsets
        
        for i in range(self.nl):
            anchors = torch.ones(1, device=targets.device)  # 简化的anchor
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            
            # 匹配目标到anchor
            t = targets * gain
            if nt:
                # 匹配
                r = t[:, :, 4:6] / anchors[:, None]  # wh比例
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # 比较
                t = t[j]  # 过滤
                
                # 偏移
                gxy = t[:, 2:4]  # 网格xy
                gxi = gain[[2, 3]] - gxy  # 反转
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
                
            # 定义
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # 网格xy
            gwh = t[:, 4:6]  # 网格wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # 网格indices
            
            # 添加
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            
        return tcls, tbox, indices, anch
    
    def balance_losses(self, lobj, lcls, lbox):
        """自动平衡损失权重"""
        # 简单的损失平衡策略
        total_loss = lobj + lcls + lbox
        if total_loss > 0:
            self.hyp['obj'] *= 0.999
            self.hyp['cls'] *= 0.999
            self.hyp['box'] *= 0.999


class FocalLoss(nn.Module):
    """Focal Loss实现"""
    
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # 必须是nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # 要求逐元素损失
        
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability
        
        # TF实现 https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """计算bbox的IoU"""
    # 获取box的坐标
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 转换xywh到xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # 交集面积
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 并集面积
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 最小外接矩形宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 最小外接矩形高度
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # 对角线距离的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心距离的平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 最小外接矩形面积
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class ComputeLoss:
    """损失计算类"""
    
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # 获取模型设备
        h = model.hyp if hasattr(model, 'hyp') else {}  # hyperparameters
        
        # 定义标准
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.get('cls_pw', 1.0)], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.get('obj_pw', 1.0)], device=device))
        
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        
        # Focal loss
        g = h.get('fl_gamma', 0.0)  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            
        det = model.model[-1] if hasattr(model, 'model') else model  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if hasattr(det, 'stride') else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k, None))
            
    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        
        # 计算损失
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp.get('box', 0.05)
        lobj *= self.hyp.get('obj', 1.0)
        lcls *= self.hyp.get('cls', 0.5)
        bs = tobj.shape[0]  # batch size
        
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """标签平滑BCE"""
    return 1.0 - 0.5 * eps, 0.5 * eps


if __name__ == '__main__':
    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    nc = 5
    
    # 模拟预测结果
    predictions = [
        torch.randn(batch_size, 1, 20, 20, nc + 5, device=device),
        torch.randn(batch_size, 1, 40, 40, nc + 5, device=device),
        torch.randn(batch_size, 1, 80, 80, nc + 5, device=device),
    ]
    
    # 模拟目标
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],  # image_id, class, x, y, w, h
        [1, 1, 0.3, 0.4, 0.1, 0.15],
    ], device=device)
    
    # 计算损失
    criterion = YOLOLoss(nc=nc, device=device)
    loss, loss_items = criterion(predictions, targets)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Box loss: {loss_items[0]:.4f}")
    print(f"Obj loss: {loss_items[1]:.4f}")
    print(f"Cls loss: {loss_items[2]:.4f}")