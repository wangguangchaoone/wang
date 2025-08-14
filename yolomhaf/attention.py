import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """Coordinate Attention (CA).
    Splits global pooling into two directions to encode positional info.
    Reference: https://arxiv.org/abs/2103.02907
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        reduced_channels = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(reduced_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = x * a_h * a_w
        return out


class ECA(nn.Module):
    """Efficient Channel Attention.
    Performs 1D conv over channel descriptor with adaptive kernel size.
    Reference: https://arxiv.org/abs/1910.03151
    """

    def __init__(self, channels: int, k_size: int = None, gamma: int = 2, b: int = 1):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y