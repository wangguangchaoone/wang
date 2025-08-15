"""
注意力机制模块
包含坐标注意力(CA)、高效通道注意力(ECA)和混合注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoordAtt(nn.Module):
    """坐标注意力机制 (Coordinate Attention)
    通过在水平和垂直方向上进行全局平均池化来捕获空间位置信息
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # 水平和垂直方向的全局平均池化
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]

        # 拼接并通过1x1卷积
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分离水平和垂直注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 应用注意力权重
        out = identity * a_h * a_w

        return out


class h_swish(nn.Module):
    """Hard Swish激活函数"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class ECA(nn.Module):
    """高效通道注意力 (Efficient Channel Attention)
    使用1D卷积实现高效的通道注意力机制
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        
        # 计算1D卷积的卷积核大小
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)  # [N, C, 1, 1]
        
        # 1D卷积，去除空间维度
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid激活
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class SEAttention(nn.Module):
    """SE注意力机制"""
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HybridAttention(nn.Module):
    """混合注意力机制
    结合坐标注意力(CA)和高效通道注意力(ECA)
    通过协同的空间定位和通道增强来提升特征表达能力
    """
    def __init__(self, channel, reduction=32, gamma=2, b=1, use_ca=True, use_eca=True):
        super(HybridAttention, self).__init__()
        
        self.use_ca = use_ca
        self.use_eca = use_eca
        
        if self.use_ca:
            self.ca = CoordAtt(channel, channel, reduction)
            
        if self.use_eca:
            self.eca = ECA(channel, gamma, b)
            
        # 特征融合权重
        if self.use_ca and self.use_eca:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            
    def forward(self, x):
        if self.use_ca and self.use_eca:
            # 同时使用CA和ECA
            ca_out = self.ca(x)
            eca_out = self.eca(x)
            
            # 加权融合
            alpha = torch.sigmoid(self.fusion_weight)
            out = alpha * ca_out + (1 - alpha) * eca_out
            
        elif self.use_ca:
            out = self.ca(x)
        elif self.use_eca:
            out = self.eca(x)
        else:
            out = x
            
        return out


class CBAM(nn.Module):
    """CBAM注意力机制 (Convolutional Block Attention Module)"""
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        
        return x


class MultiScaleAttention(nn.Module):
    """多尺度注意力机制"""
    def __init__(self, channel, scales=[1, 3, 5]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(channel, channel, k, padding=k//2, groups=channel)
            for k in scales
        ])
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel * len(scales), channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        features = []
        for conv in self.convs:
            features.append(conv(x))
        
        # 特征融合
        multi_scale = torch.cat(features, dim=1)
        attention = self.fc(multi_scale)
        
        return x * attention