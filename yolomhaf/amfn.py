from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct
from .bifpn import BiFPNBlock, SeparableConv
from .carafe import CARAFE


class DownsampleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, kernel_size=3, stride=2, padding=1, groups=c_in, bias=False)
        self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class AMFN(nn.Module):
    """Adaptive Multi-path Feature Fusion Network (AMFN).
    - Lateral 1x1 to unify channels
    - Top-down path with CARAFE upsampling and fusion
    - Bottom-up path with depthwise downsampling and fusion
    - Final refinement using a BiFPN block (or more)
    """

    def __init__(self, c3: int, c4: int, c5: int, out_channels: int = 128, num_bifpn: int = 1):
        super().__init__()
        self.lateral3 = ConvBNAct(c3, out_channels, k=1, s=1)
        self.lateral4 = ConvBNAct(c4, out_channels, k=1, s=1)
        self.lateral5 = ConvBNAct(c5, out_channels, k=1, s=1)

        self.up5 = CARAFE(out_channels, c_mid=64, scale=2, kernel_size=5)
        self.up4 = CARAFE(out_channels, c_mid=64, scale=2, kernel_size=5)

        self.fuse4 = SeparableConv(out_channels, out_channels)
        self.fuse3 = SeparableConv(out_channels, out_channels)

        self.down3 = DownsampleConv(out_channels, out_channels)
        self.fuse4_bu = SeparableConv(out_channels, out_channels)

        self.down4 = DownsampleConv(out_channels, out_channels)
        self.fuse5_bu = SeparableConv(out_channels, out_channels)

        self.bifpns = nn.ModuleList([BiFPNBlock(out_channels) for _ in range(num_bifpn)])

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> List[torch.Tensor]:
        p3 = self.lateral3(c3)
        p4 = self.lateral4(c4)
        p5 = self.lateral5(c5)

        # Top-down with CARAFE
        p5_up = self.up5(p5)
        p4_td = self.fuse4(p4 + p5_up)
        p4_up = self.up4(p4_td)
        p3_td = self.fuse3(p3 + p4_up)

        # Bottom-up reinforcement
        p3_down = self.down3(p3_td)
        p4_bu = self.fuse4_bu(p4_td + p3_down)
        p4_down = self.down4(p4_bu)
        p5_bu = self.fuse5_bu(p5 + p4_down)

        p3_out, p4_out, p5_out = p3_td, p4_bu, p5_bu
        for bifpn in self.bifpns:
            p3_out, p4_out, p5_out = bifpn(p3_out, p4_out, p5_out)

        return [p3_out, p4_out, p5_out]