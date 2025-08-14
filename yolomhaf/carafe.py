from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE) upsampling.
    Simplified PyTorch implementation.
    """

    def __init__(self, c_in: int, c_mid: int = 64, scale: int = 2, kernel_size: int = 5):
        super().__init__()
        assert scale >= 2 and kernel_size % 2 == 1
        self.scale = scale
        self.kernel_size = kernel_size
        self.comp = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1, padding=0)
        self.enc = nn.Conv2d(
            c_mid, (kernel_size * kernel_size) * (scale * scale), kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        k = self.kernel_size
        s = self.scale

        kernel = self.enc(F.relu(self.comp(x)))  # [N, K*K*S*S, H, W]
        kernel = kernel.view(n, s * s, k * k, h * w)
        kernel = torch.softmax(kernel, dim=2)

        x_unfold = F.unfold(x, kernel_size=k, padding=k // 2)  # [N, C*K*K, H*W]
        x_unfold = x_unfold.view(n, c, k * k, h * w)

        out = torch.einsum("nckm,nskm->ncsm", x_unfold, kernel)
        out = out.view(n, c, s, s, h, w).permute(0, 1, 4, 2, 5, 3).contiguous().view(n, c, h * s, w * s)
        return out