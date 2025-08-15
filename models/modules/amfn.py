"""
自适应多路径特征融合网络(AMFN)实现
融合BiFPN的双向连接机制和CARAFE的内容感知重组技术
用以替代YOLO v8n原有的颈部网络，实现对不同尺度和复杂度特征的智能化融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class CARAFE(nn.Module):
    """CARAFE (Content-Aware ReAssembly of FEatures)
    内容感知特征重组模块，用于自适应上采样
    """
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(CARAFE, self).__init__()
        self.scale = scale
        self.k_up = k_up
        self.k_enc = k_enc
        
        # 内容编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(c, c_mid, k_enc, padding=k_enc//2),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, self.scale * self.scale * self.k_up * self.k_up, 1)
        )
        
        # 内容感知重组
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up//2*scale)

    def forward(self, x):
        N, C, H, W = x.size()
        
        # 生成重组核
        kernel = self.encoder(x)  # [N, scale*scale*k_up*k_up, H, W]
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.view(N, self.scale*self.scale, self.k_up*self.k_up, H, W)
        
        # 上采样输入特征
        x_up = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        N, C, H_up, W_up = x_up.size()
        
        # 展开特征用于重组
        x_unfold = self.unfold(x_up)  # [N, C*k_up*k_up, H*W]
        x_unfold = x_unfold.view(N, C, self.k_up*self.k_up, H, W)
        
        # 内容感知重组
        out = (x_unfold * kernel.unsqueeze(1)).sum(dim=2)  # [N, C, H, W]
        
        return out


class BiFPNLayer(nn.Module):
    """BiFPN层实现
    双向特征金字塔网络层，支持自适应权重融合
    """
    def __init__(self, channels, num_levels=3, conv_cfg=None, norm_cfg=None):
        super(BiFPNLayer, self).__init__()
        self.num_levels = num_levels
        
        # 权重参数（可学习）
        self.w1 = nn.Parameter(torch.ones(num_levels, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(num_levels - 1, dtype=torch.float32))
        self.epsilon = 1e-4
        
        # 卷积层
        self.conv = nn.ModuleList([
            Conv(channels, channels, 3, 1) for _ in range(num_levels + num_levels - 1)
        ])
        
    def forward(self, inputs):
        """
        Args:
            inputs: list of feature maps from different levels
        Returns:
            list of fused feature maps
        """
        assert len(inputs) == self.num_levels
        
        # 自上而下路径
        top_down_path = [inputs[-1]]  # 从最高层开始
        for i in range(self.num_levels - 2, -1, -1):
            # 上采样
            top_down = F.interpolate(top_down_path[-1], size=inputs[i].shape[2:], mode='nearest')
            # 加权融合
            w1 = F.relu(self.w1[i])
            w2 = F.relu(self.w1[i+1])
            fused = (w1 * inputs[i] + w2 * top_down) / (w1 + w2 + self.epsilon)
            fused = self.conv[self.num_levels - 1 - i](fused)
            top_down_path.append(fused)
        
        top_down_path.reverse()
        
        # 自下而上路径
        out = [top_down_path[0]]
        for i in range(1, self.num_levels):
            # 下采样
            bottom_up = F.interpolate(out[-1], size=top_down_path[i].shape[2:], mode='nearest')
            if bottom_up.shape[2:] != top_down_path[i].shape[2:]:
                # 如果尺寸不匹配，使用stride卷积
                bottom_up = F.adaptive_avg_pool2d(bottom_up, top_down_path[i].shape[2:])
                
            # 加权融合（三路输入）
            w1 = F.relu(self.w2[i-1])  # bottom_up权重
            w2 = F.relu(self.w1[i])    # 原始输入权重
            w3 = F.relu(self.w1[i])    # top_down权重
            
            fused = (w1 * bottom_up + w2 * inputs[i] + w3 * top_down_path[i]) / (w1 + w2 + w3 + self.epsilon)
            fused = self.conv[self.num_levels + i - 1](fused)
            out.append(fused)
            
        return out


class BiFPN(nn.Module):
    """BiFPN (Bidirectional Feature Pyramid Network)
    双向特征金字塔网络，支持多层级特征融合
    """
    def __init__(self, in_channels, out_channels, num_levels=3, num_layers=2):
        super(BiFPN, self).__init__()
        self.num_levels = num_levels
        self.num_layers = num_layers
        
        # 输入通道调整
        self.lateral_convs = nn.ModuleList([
            Conv(in_channels[i] if isinstance(in_channels, list) else in_channels, 
                 out_channels, 1, 1) 
            for i in range(num_levels)
        ])
        
        # BiFPN层
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, num_levels) 
            for _ in range(num_layers)
        ])
        
    def forward(self, inputs):
        # 通道数调整
        feats = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # 多层BiFPN处理
        for bifpn_layer in self.bifpn_layers:
            feats = bifpn_layer(feats)
            
        return feats


class AdaptiveFusion(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, channels, num_inputs=2):
        super(AdaptiveFusion, self).__init__()
        self.num_inputs = num_inputs
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_inputs, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * num_inputs, 1),
            nn.Sigmoid()
        )
        
        # 特征融合卷积
        self.fusion_conv = Conv(channels * num_inputs, channels, 1, 1)
        
    def forward(self, *inputs):
        # 拼接输入特征
        concat_feat = torch.cat(inputs, dim=1)
        
        # 生成注意力权重
        attention = self.attention(concat_feat)
        
        # 应用注意力权重
        weighted_feat = concat_feat * attention
        
        # 特征融合
        fused_feat = self.fusion_conv(weighted_feat)
        
        return fused_feat


class AMFN(nn.Module):
    """自适应多路径特征融合网络 (Adaptive Multi-path Feature fusion Network)
    融合BiFPN的双向连接机制和CARAFE的内容感知重组技术
    """
    def __init__(self, in_channels, out_channels, num_levels=3, num_bifpn_layers=2, 
                 use_carafe=True, carafe_scale=2):
        super(AMFN, self).__init__()
        self.num_levels = num_levels
        self.use_carafe = use_carafe
        
        # BiFPN特征融合
        self.bifpn = BiFPN(in_channels, out_channels, num_levels, num_bifpn_layers)
        
        # CARAFE上采样模块
        if self.use_carafe:
            self.carafe_modules = nn.ModuleList([
                CARAFE(out_channels, scale=carafe_scale) 
                for _ in range(num_levels - 1)
            ])
        
        # 自适应融合模块
        self.adaptive_fusion = nn.ModuleList([
            AdaptiveFusion(out_channels, 2) 
            for _ in range(num_levels - 1)
        ])
        
        # 输出卷积
        self.out_convs = nn.ModuleList([
            Conv(out_channels, out_channels, 3, 1) 
            for _ in range(num_levels)
        ])
        
    def forward(self, inputs):
        """
        Args:
            inputs: list of feature maps [P3, P4, P5, ...]
        Returns:
            list of enhanced feature maps
        """
        # BiFPN特征融合
        bifpn_feats = self.bifpn(inputs)
        
        # CARAFE增强和自适应融合
        enhanced_feats = []
        
        for i in range(self.num_levels):
            if i == 0:
                # 最小尺度特征直接输出
                enhanced_feats.append(self.out_convs[i](bifpn_feats[i]))
            else:
                # 使用CARAFE上采样前一层特征
                if self.use_carafe:
                    upsampled = self.carafe_modules[i-1](enhanced_feats[-1])
                else:
                    upsampled = F.interpolate(enhanced_feats[-1], 
                                            size=bifpn_feats[i].shape[2:], 
                                            mode='nearest')
                
                # 自适应融合
                fused = self.adaptive_fusion[i-1](bifpn_feats[i], upsampled)
                enhanced_feats.append(self.out_convs[i](fused))
                
        return enhanced_feats


class AMFN_Lite(nn.Module):
    """轻量化版本的AMFN"""
    def __init__(self, in_channels, out_channels, num_levels=3):
        super(AMFN_Lite, self).__init__()
        self.num_levels = num_levels
        
        # 简化的特征融合
        self.lateral_convs = nn.ModuleList([
            Conv(in_channels[i] if isinstance(in_channels, list) else in_channels, 
                 out_channels, 1, 1) 
            for i in range(num_levels)
        ])
        
        # 轻量化融合模块
        self.fusion_convs = nn.ModuleList([
            Conv(out_channels * 2, out_channels, 3, 1) 
            for _ in range(num_levels - 1)
        ])
        
    def forward(self, inputs):
        # 通道调整
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        
        # 自上而下融合
        for i in range(len(feats) - 2, -1, -1):
            # 上采样高层特征
            upsampled = F.interpolate(feats[i+1], size=feats[i].shape[2:], mode='nearest')
            # 融合
            feats[i] = self.fusion_convs[i](torch.cat([feats[i], upsampled], dim=1))
            
        return feats