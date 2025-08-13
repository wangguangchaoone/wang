import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional

from .common import ConvBNAct, DepthwiseSeparableConv, SqueezeExcite


class DPHC(nn.Module):
    """Detail-Preserving High-Pass Convolution: depthwise-dilated branch + learnable LoG branch + channel gating."""
    def __init__(self, channels: int, dilations: Tuple[int, int] = (1, 2)):
        super().__init__()
        self.branch_dw1 = DepthwiseSeparableConv(channels, channels, 3, stride=1, dilation=dilations[0])
        self.branch_dw2 = DepthwiseSeparableConv(channels, channels, 3, stride=1, dilation=dilations[1])
        # Learnable LoG-like depthwise filter
        self.log = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self._init_log_kernel()
        # Channel gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(channels * 2, channels, 1)

    def _init_log_kernel(self):
        # 3x3 LoG kernel (discrete) normalized to zero-sum
        k = torch.tensor([[0., -1., 0.],
                          [-1., 4., -1.],
                          [0., -1., 0.]])
        k = k.unsqueeze(0).unsqueeze(0)  # 1,1,3,3
        with torch.no_grad():
            w = k.repeat(self.log.weight.shape[0], 1, 1, 1)  # C,1,3,3
            self.log.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_dw1(x)
        b2 = self.branch_dw2(x)
        hp = self.log(x)
        # High-pass enhanced branch
        enh = b1 + b2 + hp
        g = self.gate(enh)
        out = self.fuse(torch.cat([x, enh * g], dim=1))
        return out


class SaliencyGuidedDynamicTokenizer(nn.Module):
    def __init__(self, channels: int, token_ratio: float = 0.12):
        super().__init__()
        self.token_ratio = token_ratio
        # contrast head to produce saliency map
        self.contrast = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, feat: torch.Tensor):
        # feat: B,C,H,W
        B, C, H, W = feat.shape
        sal = self.contrast(feat)  # B,1,H,W
        sal = torch.sigmoid(sal)
        num_tokens = max(1, int(self.token_ratio * H * W))
        sal_flat = sal.view(B, -1)
        topk = torch.topk(sal_flat, k=num_tokens, dim=1)
        indices = topk.indices  # B, K
        # gather tokens
        feat_flat = feat.view(B, C, -1).transpose(1, 2)  # B,HW,C
        tokens = torch.gather(feat_flat, 1, indices.unsqueeze(-1).expand(-1, -1, C))  # B,K,C
        return tokens, indices, sal


class WindowSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "Feature map must be divisible by window size"
        x_windows = rearrange(x, 'b c (h p) (w q) -> (b h w) (p q) c', p=ws, q=ws)  # nW, N, C
        qkv = self.qkv(x_windows)
        q, k, v = qkv.chunk(3, dim=-1)  # nW, N, C
        # reshape to heads
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5
        q = rearrange(q, 'n n_tok (h d) -> n h n_tok d', h=self.num_heads)
        k = rearrange(k, 'n n_tok (h d) -> n h n_tok d', h=self.num_heads)
        v = rearrange(v, 'n n_tok (h d) -> n h n_tok d', h=self.num_heads)
        attn = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)  # n, h, N, N
        out = attn @ v  # n, h, N, d
        out = rearrange(out, 'n h n_tok d -> n n_tok (h d)')
        out = self.proj(out)
        out = rearrange(out, '(b h w) (p q) c -> b c (h p) (w q)', b=B, h=H // ws, w=W // ws, p=ws, q=ws)
        return out


class GlobalMemoryCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_memory: int = 16):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        # Learned memory tokens
        self.mem = nn.Parameter(torch.randn(1, num_memory, channels) * 0.02)
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: B,K,C  queries attend to memory
        B, K, C = tokens.shape
        q = self.to_q(tokens)
        kv = self.to_kv(self.mem.expand(B, -1, -1))
        k, v = kv.chunk(2, dim=-1)
        # heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class SemanticGatedFusion(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        self.lproj = ConvBNAct(low_channels, out_channels, 1)
        self.hproj = ConvBNAct(high_channels, out_channels, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        # upsample high to low
        high = F.interpolate(self.hproj(high), size=low.shape[-2:], mode='bilinear', align_corners=False)
        low = self.lproj(low)
        gate = self.gate(high)
        fused = low * gate + high
        return fused


class SobelConv(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag


class DynamicQueryRefine(nn.Module):
    def __init__(self, channels: int, k: int = 128, window: int = 5, num_heads: int = 4):
        super().__init__()
        self.k = k
        self.window = window
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, logits: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # logits: B,1,H,W    feat: B,C,H,W
        B, _, H, W = logits.shape
        K = min(self.k, H * W)
        flat = logits.view(B, -1)
        topk = torch.topk(flat, k=K, dim=1)
        idx = topk.indices  # B,K
        C = feat.shape[1]
        feat_flat = feat.view(B, C, -1).transpose(1, 2)  # B,HW,C
        tokens = torch.gather(feat_flat, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # B,K,C
        # self-attn among tokens (lightweight)
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)  # B,K,C
        # project back to logits delta via 1x1 conv proxy
        delta = out.mean(dim=-1)  # B,K
        new_flat = flat.clone()
        for b in range(B):
            new_flat[b, idx[b]] = new_flat[b, idx[b]] + delta[b]
        return new_flat.view(B, 1, H, W)