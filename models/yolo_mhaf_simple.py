"""
简化版YOLO-MHAF模型
基于改进YOLO v8n的多头自适应融合检测算法
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional

from .modules.ghost import GhostConv, C2f_Ghost
from .modules.spd_conv import SPDConv, C2f_SPD
from .modules.attention import HybridAttention
from .modules.c2f_hybrid import C2f_Hybrid, SPPF_Hybrid
from .modules.amfn import AMFN_Lite


class Conv(nn.Module):
    """标准卷积层"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Detect(nn.Module):
    """YOLO检测头"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.no = nc + 5  # 每个anchor的输出数
        self.nl = len(ch)  # 检测层数
        self.na = 1  # anchor数量
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            out = self.m[i](x[i])
            bs, _, ny, nx = out.shape
            out = out.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            z.append(out)
        return z


class YOLOMHAFSimple(nn.Module):
    """简化版YOLO-MHAF模型"""
    
    def __init__(self, nc=5, ch=3, width_mult=0.25, depth_mult=0.33):
        super().__init__()
        self.nc = nc
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        
        # 计算通道数
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        c1 = make_divisible(64 * width_mult)
        c2 = make_divisible(128 * width_mult)
        c3 = make_divisible(256 * width_mult)
        c4 = make_divisible(512 * width_mult)
        c5 = make_divisible(1024 * width_mult)
        
        # 骨干网络 - 混合轻量化策略
        self.backbone = nn.ModuleList([
            # Stage 1 - 浅层使用标准卷积
            Conv(ch, c1, 6, 2, 2),  # P1/2
            Conv(c1, c2, 3, 2),     # P2/4
            
            # Stage 2 - 浅层使用Ghost模块
            C2f_Ghost(c2, c2, max(round(3 * depth_mult), 1), True),
            Conv(c2, c3, 3, 2),     # P3/8
            C2f_Ghost(c3, c3, max(round(6 * depth_mult), 1), True),
            
            # Stage 3 - 深层使用SPD-Conv
            SPDConv(c3, c4, 3, 2),  # P4/16
            C2f_SPD(c4, c4, max(round(6 * depth_mult), 1), True),
            SPDConv(c4, c5, 3, 2),  # P5/32
            C2f_SPD(c5, c5, max(round(3 * depth_mult), 1), True),
            
            # SPPF with Hybrid Attention
            SPPF_Hybrid(c5, c5, 5, True, 'hybrid')
        ])
        
        # 颈部网络 - 特征融合
        self.neck = nn.ModuleList([
            # 上采样路径
            nn.Upsample(None, 2, 'nearest'),
            C2f_Hybrid(c5 + c4, c4, max(round(3 * depth_mult), 1), False, 1, 0.5, True, 'hybrid', 'global'),
            
            nn.Upsample(None, 2, 'nearest'),  
            C2f_Hybrid(c4 + c3, c3, max(round(3 * depth_mult), 1), False, 1, 0.5, True, 'hybrid', 'global'),
            
            # 下采样路径
            Conv(c3, c3, 3, 2),
            C2f_Hybrid(c3 + c4, c4, max(round(3 * depth_mult), 1), False, 1, 0.5, True, 'hybrid', 'global'),
            
            Conv(c4, c4, 3, 2),
            C2f_Hybrid(c4 + c5, c5, max(round(3 * depth_mult), 1), False, 1, 0.5, True, 'hybrid', 'global'),
        ])
        
        # 检测头
        self.head = Detect(nc, [c3, c4, c5])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 骨干网络特征提取
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:  # 保存P3, P4, P5特征
                features.append(x)
        
        # P5, P4, P3
        p5, p4, p3 = features[2], features[1], features[0]
        
        # 颈部网络特征融合
        # 上采样路径
        x = self.neck[0](p5)  # 上采样P5
        x = torch.cat([x, p4], 1)  # 与P4拼接
        x = self.neck[1](x)  # C2f_Hybrid处理
        p4_out = x
        
        x = self.neck[2](x)  # 上采样
        x = torch.cat([x, p3], 1)  # 与P3拼接
        x = self.neck[3](x)  # C2f_Hybrid处理
        p3_out = x
        
        # 下采样路径
        x = self.neck[4](p3_out)  # 下采样
        x = torch.cat([x, p4_out], 1)  # 与P4拼接
        x = self.neck[5](x)  # C2f_Hybrid处理
        p4_final = x
        
        x = self.neck[6](x)  # 下采样
        x = torch.cat([x, p5], 1)  # 与P5拼接
        x = self.neck[7](x)  # C2f_Hybrid处理
        p5_final = x
        
        # 检测头
        outputs = self.head([p3_out, p4_final, p5_final])
        
        return outputs
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # 假设float32
        }


def create_yolo_mhaf_n(nc=5, pretrained=False):
    """创建YOLO-MHAF-n模型"""
    model = YOLOMHAFSimple(nc=nc, width_mult=0.25, depth_mult=0.33)
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


def create_yolo_mhaf_s(nc=5, pretrained=False):
    """创建YOLO-MHAF-s模型"""
    model = YOLOMHAFSimple(nc=nc, width_mult=0.50, depth_mult=0.33)
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


def create_yolo_mhaf_m(nc=5, pretrained=False):
    """创建YOLO-MHAF-m模型"""
    model = YOLOMHAFSimple(nc=nc, width_mult=0.75, depth_mult=0.67)
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_yolo_mhaf_n(nc=5)
    model.eval()
    
    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(x)
    
    print("YOLO-MHAF Model Test:")
    print(f"Input shape: {x.shape}")
    print(f"Number of detection layers: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
    
    # 模型信息
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")