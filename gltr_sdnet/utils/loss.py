import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from ..models.layers import SobelConv


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
        loss = alpha_factor * loss
    return loss.mean()


def soft_iou_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = (probs + targets - probs * targets).sum(dim=(1, 2, 3)) + 1e-6
    iou = intersection / union
    return (1 - iou).mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelConv()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pred_edge = self.sobel(probs)
        gt_edge = self.sobel(targets)
        return F.l1_loss(pred_edge, gt_edge)


def background_weight_map(images: torch.Tensor) -> torch.Tensor:
    # images: B,1,H,W in [0,1]
    # emphasize high-texture regions via Laplacian magnitude
    k = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], device=images.device).view(1, 1, 3, 3)
    lap = F.conv2d(images, k, padding=1)
    w = torch.relu(lap).abs()
    w = w / (w.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return torch.clamp(w, 0.5, 3.0)


def background_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    weights = background_weight_map(images)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (bce * weights).mean()


class GLTRLoss(nn.Module):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 lambda_lovasz: float = 1.0, lambda_boundary: float = 1.0, lambda_bg: float = 0.5):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lambda_lovasz = lambda_lovasz
        self.lambda_boundary = lambda_boundary
        self.lambda_bg = lambda_bg
        self.boundary_loss = BoundaryLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs['logits']
        targets = batch['mask']
        images = batch['image']
        lfocal = focal_bce_with_logits(logits, targets, alpha=self.focal_alpha, gamma=self.focal_gamma)
        l_iou = soft_iou_loss(logits, targets) * self.lambda_lovasz
        l_b = self.boundary_loss(logits, targets) * self.lambda_boundary
        l_bg = background_weighted_bce(logits, targets, images) * self.lambda_bg
        loss = lfocal + l_iou + l_b + l_bg
        return {
            'loss': loss,
            'lfocal': lfocal.detach(),
            'liou': (l_iou / max(self.lambda_lovasz, 1e-6)).detach(),
            'lboundary': (l_b / max(self.lambda_boundary, 1e-6)).detach(),
            'lbg': (l_bg / max(self.lambda_bg, 1e-6)).detach(),
        }