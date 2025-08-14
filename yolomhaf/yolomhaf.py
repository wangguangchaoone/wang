from typing import List, Tuple

import torch
import torch.nn as nn

from .backbone import Backbone
from .amfn import AMFN
from .head import DetectHead


class YOLOMHAF(nn.Module):
    def __init__(self, num_classes: int = 5, neck_channels: int = 128, num_bifpn: int = 1):
        super().__init__()
        self.backbone = Backbone()
        # Backbone outputs: C3=128, C4=256, C5=256
        self.neck = AMFN(c3=128, c4=256, c5=256, out_channels=neck_channels, num_bifpn=num_bifpn)
        self.head = DetectHead(in_channels=[neck_channels, neck_channels, neck_channels], mid_channels=neck_channels,
                               num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        preds = self.head([p3, p4, p5])
        return preds