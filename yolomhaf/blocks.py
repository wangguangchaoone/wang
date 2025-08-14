from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CoordinateAttention, ECA


class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, p: int = None, g: int = 1, act: bool = True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GhostConv(nn.Module):
    """Ghost module style conv: primary conv + cheap depthwise to generate more features."""

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, ratio: int = 2):
        super().__init__()
        c_primary = int(round(c_out / ratio))
        c_cheap = c_out - c_primary
        self.primary = ConvBNAct(c_in, c_primary, k=k, s=s)
        self.cheap = ConvBNAct(c_primary, c_cheap, k=3, s=1, p=1, g=c_primary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)


class SPDConv(nn.Module):
    """SPD-Conv: Space-to-Depth then standard conv to preserve info during downsampling."""

    def __init__(self, c_in: int, c_out: int, s: int = 2, k: int = 3):
        super().__init__()
        assert s in (2, 4), "SPDConv supports stride 2 or 4"
        self.s = s
        self.conv = ConvBNAct(c_in * (s * s), c_out, k=k, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.s
        x = x.view(b, c, h // s, s, w // s, s).permute(0, 1, 3, 5, 2, 4).reshape(b, c * s * s, h // s, w // s)
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_hidden, c_out, 3, 1, g=g)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """YOLOv8 C2f block: split and fuse multiple bottlenecks."""

    def __init__(self, c_in: int, c_out: int, n: int = 2, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, 2 * c_hidden, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * c_hidden, c_out, 1)
        self.m = nn.ModuleList([Bottleneck(c_hidden, c_hidden, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        x = self.cv1(x)
        x1, x2 = x.split(x.shape[1] // 2, dim=1)
        y.append(x1)
        y.append(x2)
        for block in self.m:
            x2 = block(x2)
            y.append(x2)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_hidden * 4, c_out, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class C2fHybrid(nn.Module):
    """C2f with hybrid attention (CA + ECA) inside each bottleneck output fusion.
    Designed to replace the C2f above SPPF to emphasize spatial and channel cues.
    """

    def __init__(self, c_in: int, c_out: int, n: int = 2, e: float = 0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, 2 * c_hidden, 1, 1)
        self.m = nn.ModuleList([Bottleneck(c_hidden, c_hidden, shortcut=True, g=1, e=1.0) for _ in range(n)])
        self.ca = CoordinateAttention(c_hidden)
        self.eca = ECA(c_hidden)
        self.cv2 = ConvBNAct((2 + n) * c_hidden, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        x = self.cv1(x)
        x1, x2 = x.split(x.shape[1] // 2, dim=1)
        parts.append(x1)
        parts.append(x2)
        for block in self.m:
            x2 = block(x2)
            x2 = self.ca(x2)
            x2 = self.eca(x2)
            parts.append(x2)
        return self.cv2(torch.cat(parts, dim=1))