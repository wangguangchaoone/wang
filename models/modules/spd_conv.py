"""
SPD-Conv模块实现
Space-to-Depth卷积，用于深层网络确保关键特征的无损传递
通过空间到深度的转换来保持特征信息的完整性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPDConv(nn.Module):
    """Space-to-Depth卷积模块
    通过空间到深度的转换来保持特征信息完整性，避免信息丢失
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, dimension=2):
        super().__init__()
        if dimension != 2:
            raise ValueError("SPDConv only supports 2D convolution")
        
        self.dimension = dimension
        
        # 计算SPD后的通道数
        if s == 2:
            # 当stride=2时，使用SPD操作
            self.spd = True
            # SPD操作会将空间尺寸减半，通道数增加4倍
            spd_c1 = c1 * 4
        else:
            self.spd = False
            spd_c1 = c1
            
        # 自动padding计算
        if p is None:
            p = k // 2
            
        self.conv = nn.Conv2d(spd_c1, c2, kernel_size=k, stride=1, 
                             padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
    def forward(self, x):
        if self.spd:
            x = self.space_to_depth(x, block_size=2)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    def space_to_depth(self, x, block_size):
        """Space-to-Depth操作
        将空间维度转换为通道维度
        """
        N, C, H, W = x.size()
        
        # 确保高度和宽度能被block_size整除
        assert H % block_size == 0 and W % block_size == 0, \
            f"Height ({H}) and Width ({W}) must be divisible by block_size ({block_size})"
        
        # 重塑张量进行space-to-depth操作
        x = x.view(N, C, H // block_size, block_size, W // block_size, block_size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(N, C * (block_size ** 2), H // block_size, W // block_size)
        
        return x


class SPDBottleneck(nn.Module):
    """使用SPD-Conv的瓶颈模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, s=1):
        super().__init__()
        c_ = int(c2 * e)
        
        # 第一个卷积使用标准卷积
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        
        # 第二个卷积使用SPD-Conv
        self.cv2 = SPDConv(c_, c2, k=k[1], s=s, g=g)
        
        # shortcut连接
        self.add = shortcut and c1 == c2 and s == 1

    def forward(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))


class C2f_SPD(nn.Module):
    """使用SPD-Conv的C2f结构
    用于深层网络，确保关键特征的无损传递
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        
        # 输入分支卷积
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )
        
        # 输出融合卷积
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # SPD瓶颈模块列表
        self.m = nn.ModuleList(SPDBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) 
                              for _ in range(n))

    def forward(self, x):
        # 分割输入特征
        y = list(self.cv1(x).chunk(2, 1))
        
        # 逐层处理
        for m in self.m:
            y.append(m(y[-1]))
            
        # 融合所有特征
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        # 另一种前向传播方式
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))