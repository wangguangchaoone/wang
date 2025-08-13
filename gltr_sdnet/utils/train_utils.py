import os
import math
import time
import yaml
import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models.model import GLTRSDNet
from .datasets import create_dataloader
from .loss import GLTRLoss
from .metrics import compute_iou, compute_f1


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]) -> GLTRSDNet:
    mcfg = cfg['model']
    model = GLTRSDNet(
        img_channels=mcfg.get('img_channels', 1),
        base_channels=mcfg.get('base_channels', 48),
        window_size=mcfg.get('window_size', 8),
        token_ratio=mcfg.get('token_ratio', 0.12),
        memory_tokens=mcfg.get('memory_tokens', 16),
        enable_dqr=mcfg.get('enable_dqr', True)
    )
    return model


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    tcfg = cfg['train']
    if tcfg.get('optimizer', 'adamw') == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=tcfg.get('lr', 3e-4), weight_decay=tcfg.get('weight_decay', 0.05))
    else:
        opt = optim.SGD(model.parameters(), lr=tcfg.get('lr', 3e-4), momentum=0.9, weight_decay=tcfg.get('weight_decay', 5e-4))
    sched = None
    if tcfg.get('cosine', True):
        sched = CosineAnnealingLR(opt, T_max=tcfg.get('epochs', 100), eta_min=tcfg.get('lr', 3e-4) * 0.1)
    return opt, sched


def train_one_epoch(model, loader, optimizer, criterion, device, print_freq=50):
    model.train()
    metrics = {'loss': 0.0}
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)
        out = model(images)
        loss_dict = criterion(out, batch={**batch, 'image': images, 'mask': targets})
        loss = loss_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in loss_dict.items():
            if k not in metrics:
                metrics[k] = 0.0
            metrics[k] += float(v)
        if (i + 1) % print_freq == 0 or (i + 1) == len(loader):
            pbar.set_description(f"loss {metrics['loss'] / (i+1):.4f}")
    for k in metrics:
        metrics[k] /= len(loader)
    return metrics


def validate(model, loader, device) -> Dict[str, float]:
    model.eval()
    iou_sum, f1_sum = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)
            outputs = model(images)
            logits = outputs['logits']
            iou_sum += compute_iou(logits, targets)
            f1_sum += compute_f1(logits, targets)
    return {'miou': iou_sum / len(loader), 'f1': f1_sum / len(loader)}


def save_checkpoint(model, optimizer, epoch, save_dir: str, best: bool = False):
    os.makedirs(os.path.join(save_dir, 'weights'), exist_ok=True)
    path = os.path.join(save_dir, 'weights', 'best.pt' if best else f'epoch_{epoch:03d}.pt')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(ckpt, path)