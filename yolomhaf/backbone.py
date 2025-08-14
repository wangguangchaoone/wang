from typing import Tuple

import torch
import torch.nn as nn

from .blocks import ConvBNAct, GhostConv, SPDConv, C2f, C2fHybrid, SPPF


class Backbone(nn.Module):
    """Backbone with mixed lightweight strategy:
    - Shallow: GhostConv
    - Deep: SPD-Conv
    - Top: SPPF + C2fHybrid

    Returns C3, C4, C5 feature maps (stride 8, 16, 32).
    """

    def __init__(self):
        super().__init__()
        # Stem and early stages
        self.stem = GhostConv(3, 32, k=3, s=2)          # 640 -> 320
        self.c2f1 = C2f(32, 32, n=1)

        self.stage2_down = GhostConv(32, 64, k=3, s=2)  # 320 -> 160
        self.c2f2 = C2f(64, 64, n=2)

        # Deeper with SPD-Conv
        self.stage3_down = SPDConv(64, 128, s=2)        # 160 -> 80  (C3)
        self.c2f3 = C2f(128, 128, n=2)

        self.stage4_down = SPDConv(128, 256, s=2)       # 80 -> 40   (C4)
        self.c2f4 = C2f(256, 256, n=2)

        self.stage5_down = SPDConv(256, 256, s=2)       # 40 -> 20   (C5 pre)
        self.sppf = SPPF(256, 256)
        self.c2f_hybrid = C2fHybrid(256, 256, n=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.c2f1(x)

        x = self.stage2_down(x)
        x = self.c2f2(x)

        x = self.stage3_down(x)
        c3 = self.c2f3(x)

        x = self.stage4_down(c3)
        c4 = self.c2f4(x)

        x = self.stage5_down(c4)
        x = self.sppf(x)
        c5 = self.c2f_hybrid(x)

        return c3, c4, c5