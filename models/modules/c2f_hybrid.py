"""
C2f-Hybrid结构实现
结合混合注意力机制的C2f模块，用于替换SPPF上层的C2f模块
通过协同的空间定位和通道增强使模型能够精准捕获花生叶片病害的多维度特征信息
"""

import torch
import torch.nn as nn
from .attention import HybridAttention, CoordAtt, ECA


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


class Bottleneck_Hybrid(nn.Module):
    """带混合注意力机制的瓶颈模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, 
                 use_attention=True, attention_type='hybrid'):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
        # 添加注意力机制
        self.use_attention = use_attention
        if self.use_attention:
            if attention_type == 'hybrid':
                self.attention = HybridAttention(c2)
            elif attention_type == 'ca':
                self.attention = CoordAtt(c2, c2)
            elif attention_type == 'eca':
                self.attention = ECA(c2)
            else:
                self.attention = None
                self.use_attention = False

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        
        # 应用注意力机制
        if self.use_attention and self.attention is not None:
            out = self.attention(out)
            
        return x + out if self.add else out


class C2f_Hybrid(nn.Module):
    """C2f-Hybrid结构
    结合混合注意力机制的C2f模块，用于替换SPPF上层的C2f模块
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, 
                 use_attention=True, attention_type='hybrid', attention_position='bottleneck'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 瓶颈模块列表
        self.m = nn.ModuleList(
            Bottleneck_Hybrid(self.c, self.c, shortcut, g, k=((3, 3)), e=1.0,
                             use_attention=use_attention and attention_position == 'bottleneck',
                             attention_type=attention_type) 
            for _ in range(n)
        )
        
        # 全局注意力机制（应用在整个模块输出上）
        self.use_global_attention = use_attention and attention_position == 'global'
        if self.use_global_attention:
            if attention_type == 'hybrid':
                self.global_attention = HybridAttention(c2)
            elif attention_type == 'ca':
                self.global_attention = CoordAtt(c2, c2)
            elif attention_type == 'eca':
                self.global_attention = ECA(c2)
            else:
                self.global_attention = None
                self.use_global_attention = False

    def forward(self, x):
        # 输入分支处理
        y = list(self.cv1(x).chunk(2, 1))
        
        # 通过瓶颈模块
        y.extend(m(y[-1]) for m in self.m)
        
        # 特征融合
        out = self.cv2(torch.cat(y, 1))
        
        # 全局注意力
        if self.use_global_attention and self.global_attention is not None:
            out = self.global_attention(out)
            
        return out

    def forward_split(self, x):
        """另一种前向传播方式"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        
        if self.use_global_attention and self.global_attention is not None:
            out = self.global_attention(out)
            
        return out


class C2f_HybridV2(nn.Module):
    """C2f-Hybrid结构的增强版本
    支持多种注意力机制的组合和自适应权重调整
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 use_ca=True, use_eca=True, use_se=False, adaptive_weight=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # 瓶颈模块
        self.m = nn.ModuleList(
            Bottleneck_Hybrid(self.c, self.c, shortcut, g, k=((3, 3)), e=1.0,
                             use_attention=False)  # 在瓶颈中不使用注意力
            for _ in range(n)
        )
        
        # 多种注意力机制
        self.attentions = nn.ModuleList()
        self.attention_types = []
        
        if use_ca:
            self.attentions.append(CoordAtt(c2, c2))
            self.attention_types.append('ca')
            
        if use_eca:
            self.attentions.append(ECA(c2))
            self.attention_types.append('eca')
            
        if use_se:
            from .attention import SEAttention
            self.attentions.append(SEAttention(c2))
            self.attention_types.append('se')
        
        # 自适应权重
        self.adaptive_weight = adaptive_weight
        if self.adaptive_weight and len(self.attentions) > 1:
            self.attention_weights = nn.Parameter(torch.ones(len(self.attentions)))
            
    def forward(self, x):
        # 基本的C2f前向传播
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        
        # 应用多种注意力机制
        if len(self.attentions) == 0:
            return out
        elif len(self.attentions) == 1:
            return self.attentions[0](out)
        else:
            # 多注意力融合
            attention_outputs = []
            for attention in self.attentions:
                attention_outputs.append(attention(out))
                
            if self.adaptive_weight:
                # 自适应权重融合
                weights = torch.softmax(self.attention_weights, dim=0)
                fused_out = sum(w * att_out for w, att_out in zip(weights, attention_outputs))
            else:
                # 平均融合
                fused_out = sum(attention_outputs) / len(attention_outputs)
                
            return fused_out


class SPPF_Hybrid(nn.Module):
    """带混合注意力的SPPF模块"""
    def __init__(self, c1, c2, k=5, use_attention=True, attention_type='hybrid'):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # 注意力机制
        self.use_attention = use_attention
        if self.use_attention:
            if attention_type == 'hybrid':
                self.attention = HybridAttention(c2)
            elif attention_type == 'ca':
                self.attention = CoordAtt(c2, c2)
            elif attention_type == 'eca':
                self.attention = ECA(c2)
            else:
                self.attention = None
                self.use_attention = False

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        
        if self.use_attention and self.attention is not None:
            out = self.attention(out)
            
        return out