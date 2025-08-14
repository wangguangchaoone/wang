from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct


class SeparableConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        p = (k - 1) // 2
        self.depthwise = nn.Conv2d(c_in, c_in, kernel_size=k, stride=s, padding=p, groups=c_in, bias=False)
        self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class BiFPNBlock(nn.Module):
    """One BiFPN iteration across P3, P4, P5 feature maps.
    Uses learnable weights (positive, normalized) for fusion.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.eps = 1e-4
        # top-down weights
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        # bottom-up weights
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32))

        self.p5_td = SeparableConv(channels, channels)
        self.p4_td = SeparableConv(channels, channels)
        self.p3_out = SeparableConv(channels, channels)

        self.p3_bu = SeparableConv(channels, channels)
        self.p4_out = SeparableConv(channels, channels)
        self.p5_out = SeparableConv(channels, channels)

    def _norm(self, w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (torch.sum(w, dim=0) + self.eps)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> List[torch.Tensor]:
        # top-down
        w1 = self._norm(self.w1)
        w2 = self._norm(self.w2)
        p5_td = self.p5_td(p5)
        p4_td = self.p4_td(w1[0] * p4 + w1[1] * F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest"))
        p3_out = self.p3_out(w2[0] * p3 + w2[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest"))

        # bottom-up
        w3 = self._norm(self.w3)
        w4 = self._norm(self.w4)
        p3_bu = self.p3_bu(p3_out)
        p4_out = self.p4_out(w3[0] * p4_td + w3[1] * F.max_pool2d(p3_bu, kernel_size=2))
        p5_out = self.p5_out(w4[0] * p5_td + w4[1] * F.max_pool2d(p4_out, kernel_size=2))
        return [p3_out, p4_out, p5_out]