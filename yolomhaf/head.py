from typing import List

import torch
import torch.nn as nn

from .blocks import ConvBNAct


class ScaleHead(nn.Module):
    def __init__(self, c_in: int, c_mid: int, num_outputs: int):
        super().__init__()
        self.conv = ConvBNAct(c_in, c_mid, k=3, s=1)
        self.pred = nn.Conv2d(c_mid, num_outputs, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pred(x)
        return x


class DetectHead(nn.Module):
    def __init__(self, in_channels: List[int], mid_channels: int, num_classes: int):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.heads = nn.ModuleList([ScaleHead(c, mid_channels, self.num_outputs) for c in in_channels])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for x, head in zip(feats, self.heads):
            outputs.append(head(x))
        return outputs