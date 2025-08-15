import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerDetectionHead(nn.Module):
    """基于Transformer的检测头"""
    def __init__(self, d_model, n_heads, d_ff, n_layers, num_classes, num_anchors=3):
        super(TransformerDetectionHead, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # 检测头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 4 * num_anchors)  # x, y, w, h
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes * num_anchors)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_anchors)
        )
        
    def forward(self, features, global_feat):
        batch_size = features[0].size(0)
        
        # 将特征图转换为序列
        feature_sequences = []
        for feat in features:
            # 将特征图重塑为序列
            b, c, h, w = feat.shape
            seq_feat = feat.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
            feature_sequences.append(seq_feat)
        
        # 合并所有特征序列
        combined_features = torch.cat(feature_sequences, dim=1)  # (B, total_H*W, C)
        
        # 添加全局特征
        global_feat = global_feat.unsqueeze(1)  # (B, 1, C)
        combined_features = torch.cat([global_feat, combined_features], dim=1)
        
        # 通过Transformer层
        for transformer_layer in self.transformer_layers:
            combined_features = transformer_layer(combined_features)
        
        # 分离全局特征和局部特征
        global_feat_enhanced = combined_features[:, 0]  # (B, C)
        local_features = combined_features[:, 1:]  # (B, total_H*W, C)
        
        # 检测头预测
        bbox_pred = self.regression_head(global_feat_enhanced)  # (B, 4*A)
        cls_pred = self.classification_head(global_feat_enhanced)  # (B, C*A)
        conf_pred = self.confidence_head(global_feat_enhanced)  # (B, A)
        
        # 重塑预测结果
        bbox_pred = bbox_pred.view(batch_size, self.num_anchors, 4)
        cls_pred = cls_pred.view(batch_size, self.num_anchors, self.num_classes)
        conf_pred = conf_pred.view(batch_size, self.num_anchors, 1)
        
        return bbox_pred, cls_pred, conf_pred

class YOLODetectionHead(nn.Module):
    """YOLO风格的检测头"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLODetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 检测头卷积层
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # 输出层
        self.reg_conv = nn.Conv2d(in_channels, num_anchors * 4, 1)  # 边界框回归
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, 1)  # 分类
        self.obj_conv = nn.Conv2d(in_channels, num_anchors * 1, 1)  # 目标置信度
        
    def forward(self, x):
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 预测输出
        bbox_pred = self.reg_conv(x)
        cls_pred = self.cls_conv(x)
        obj_pred = self.obj_conv(x)
        
        return bbox_pred, cls_pred, obj_pred

class InfraredDetectionModel(nn.Module):
    """完整的红外弱小目标检测模型"""
    def __init__(self, num_classes=1, d_model=256, n_heads=8, n_layers=6):
        super(InfraredDetectionModel, self).__init__()
        
        # 骨干网络
        self.backbone = InfraredBackbone(
            input_channels=1,  # 红外图像通常是单通道
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 64, 128, 256, 512],
            out_channels=d_model
        )
        
        # Transformer检测头
        self.transformer_head = TransformerDetectionHead(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            n_layers=n_layers,
            num_classes=num_classes
        )
        
        # YOLO检测头（用于多尺度检测）
        self.yolo_heads = nn.ModuleList([
            YOLODetectionHead(d_model, num_classes)
            for _ in range(3)  # 3个不同尺度的检测头
        ])
        
        # 特征融合
        self.feature_fusion = nn.ModuleList([
            nn.Conv2d(d_model, d_model, 1)
            for _ in range(3)
        ])
        
    def forward(self, x):
        # 骨干网络特征提取
        conv_features, global_feat = self.backbone(x)
        
        # FPN特征融合
        fpn_features = self.fpn(conv_features)
        
        # 选择3个不同尺度的特征
        selected_features = [fpn_features[1], fpn_features[2], fpn_features[3]]
        
        # Transformer检测头
        transformer_bbox, transformer_cls, transformer_conf = self.transformer_head(
            selected_features, global_feat
        )
        
        # YOLO检测头
        yolo_outputs = []
        for i, (head, feature, fusion) in enumerate(zip(self.yolo_heads, selected_features, self.feature_fusion)):
            fused_feature = fusion(feature)
            bbox, cls, obj = head(fused_feature)
            yolo_outputs.append((bbox, cls, obj))
        
        return {
            'transformer': (transformer_bbox, transformer_cls, transformer_conf),
            'yolo': yolo_outputs,
            'features': selected_features,
            'global_feat': global_feat
        }

class LossFunction(nn.Module):
    """损失函数"""
    def __init__(self, num_classes=1, anchors=3):
        super(LossFunction, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        
        # 损失权重
        self.bbox_weight = 1.0
        self.cls_weight = 1.0
        self.obj_weight = 1.0
        
        # 损失函数
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # 这里简化处理，实际应该根据targets计算损失
        # 返回总损失
        total_loss = 0.0
        
        # Transformer检测头损失
        transformer_bbox, transformer_cls, transformer_conf = predictions['transformer']
        
        # YOLO检测头损失
        yolo_outputs = predictions['yolo']
        
        # 计算各种损失（这里简化）
        bbox_loss = torch.mean(transformer_bbox ** 2)
        cls_loss = torch.mean(transformer_cls ** 2)
        conf_loss = torch.mean(transformer_conf ** 2)
        
        total_loss = self.bbox_weight * bbox_loss + \
                    self.cls_weight * cls_loss + \
                    self.obj_weight * conf_loss
        
        return total_loss, {
            'bbox_loss': bbox_loss.item(),
            'cls_loss': cls_loss.item(),
            'conf_loss': conf_loss.item(),
            'total_loss': total_loss.item()
        }