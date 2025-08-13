import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = None, groups: int = 1, act: str = "silu"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                             dilation=dilation, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.act(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, use_se: bool = True):
        super().__init__()
        hidden = channels * expansion // 2
        self.conv1 = ConvBNAct(channels, hidden, 1)
        self.conv2 = ConvBNAct(hidden, channels, 3)
        self.se = SqueezeExcite(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + identity


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBNAct(in_channels, out_channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)