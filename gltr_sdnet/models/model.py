import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .common import ConvBNAct, ResidualBlock, Downsample
from .layers import DPHC, SaliencyGuidedDynamicTokenizer, WindowSelfAttention, GlobalMemoryCrossAttention, SemanticGatedFusion, DynamicQueryRefine


class SimpleBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 48):
        super().__init__()
        # stem to 1/2
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels // 2, 3, stride=2),
            ResidualBlock(base_channels // 2),
        )
        # 1/4
        self.down1 = Downsample(base_channels // 2, base_channels)
        self.block1 = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
        )
        # 1/8
        self.down2 = Downsample(base_channels, base_channels * 2)
        self.block2 = nn.Sequential(
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
        )
        # 1/16
        self.down3 = Downsample(base_channels * 2, base_channels * 4)
        self.block3 = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

    def forward(self, x: torch.Tensor):
        x2 = self.stem(x)            # 1/2
        x4 = self.block1(self.down1(x2))   # 1/4
        x8 = self.block2(self.down2(x4))   # 1/8
        x16 = self.block3(self.down3(x8))  # 1/16
        return x4, x8, x16


class GLTRSDNet(nn.Module):
    def __init__(self, img_channels: int = 1, base_channels: int = 48, window_size: int = 8,
                 token_ratio: float = 0.12, memory_tokens: int = 16, enable_dqr: bool = True):
        super().__init__()
        self.backbone = SimpleBackbone(img_channels, base_channels)
        c4, c8, c16 = base_channels, base_channels * 2, base_channels * 4
        self.dphc4 = DPHC(c4)
        self.dphc8 = DPHC(c8)
        self.tokenizer = SaliencyGuidedDynamicTokenizer(c8, token_ratio)
        self.wsa8 = WindowSelfAttention(c8, num_heads=4, window_size=window_size)
        self.mem_ca = GlobalMemoryCrossAttention(c8, num_heads=4, num_memory=memory_tokens)
        # decoder with SGF
        self.sgf8 = SemanticGatedFusion(low_channels=c8, high_channels=c16, out_channels=c8)
        self.sgf4 = SemanticGatedFusion(low_channels=c4, high_channels=c8, out_channels=c4)
        self.head = nn.Sequential(
            ConvBNAct(c4, c4, 3),
            nn.Conv2d(c4, 1, 1)
        )
        self.enable_dqr = enable_dqr
        if enable_dqr:
            self.dqr = DynamicQueryRefine(channels=c4, k=256, window=5, num_heads=4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        x4, x8, x16 = self.backbone(x)
        x4 = self.dphc4(x4)
        x8 = self.dphc8(x8)
        # window self-attention on x8
        # ensure divisibility by window
        ws = self.wsa8.window_size
        def pad_to_window(t: torch.Tensor, ws: int):
            h, w = t.shape[-2:]
            ph = (ws - h % ws) % ws
            pw = (ws - w % ws) % ws
            if ph or pw:
                t = F.pad(t, (0, pw, 0, ph), value=0)
            return t, ph, pw

        x8_pad, ph, pw = pad_to_window(x8, ws=ws)
        x8_attn = self.wsa8(x8_pad)
        if ph or pw:
            x8_attn = x8_attn[..., :x8.shape[-2], :x8.shape[-1]]
        # dynamic tokens from x8
        tokens, indices, sal = self.tokenizer(x8_attn)
        mem_out = self.mem_ca(tokens)  # B,K,C
        # scatter token updates back to feature map without in-place on x8_flat
        x8_flat = x8_attn.view(B, x8_attn.shape[1], -1).transpose(1, 2)  # B,HW,C
        updates = torch.zeros_like(x8_flat)
        # scatter_add along HW dimension
        updates = updates.scatter_add(1, indices.unsqueeze(-1).expand_as(mem_out), mem_out)
        x8_updated = (x8_flat + updates).transpose(1, 2).view_as(x8_attn)
        # decoder
        d8 = self.sgf8(x8_updated, x16)
        d4 = self.sgf4(x4, d8)
        logits = self.head(d4)
        # upsample to input
        logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        if self.enable_dqr:
            logits_up = self.dqr(logits_up, F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=False))
        return {"logits": logits_up, "saliency": F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)}