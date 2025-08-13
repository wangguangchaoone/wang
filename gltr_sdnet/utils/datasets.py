import os
from typing import Tuple, List, Optional
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0)  # 1,H,W


class IRTargetDataset(Dataset):
    def __init__(self, root: str, img_size: int = 512, split: str = 'train', augment: bool = True):
        super().__init__()
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        self.names = sorted([n for n in os.listdir(self.img_dir) if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.img_size = img_size
        self.split = split
        self.augment = augment and split == 'train'

    def __len__(self):
        return len(self.names)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img = np.ascontiguousarray(np.fliplr(img))
            mask = np.ascontiguousarray(np.fliplr(mask))
        # random scale and crop to img_size
        H, W = img.shape
        scale = random.uniform(0.9, 1.1)
        newH, newW = int(H * scale), int(W * scale)
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (newW, newH), interpolation=cv2.INTER_NEAREST)
        if newH < self.img_size or newW < self.img_size:
            pad_h = max(0, self.img_size - newH)
            pad_w = max(0, self.img_size - newW)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            newH, newW = img.shape
        y = random.randint(0, newH - self.img_size)
        x = random.randint(0, newW - self.img_size)
        img = img[y:y+self.img_size, x:x+self.img_size]
        mask = mask[y:y+self.img_size, x:x+self.img_size]
        # gamma
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            img = np.clip((img / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')
        img = _read_gray(img_path)
        mask = _read_gray(mask_path)
        mask = (mask > 127).astype(np.uint8) * 255
        if self.augment:
            img, mask = self._augment(img, mask)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        image_t = _to_tensor(img)
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return {
            'image': image_t,
            'mask': mask_t,
            'name': name,
        }


class SyntheticIrSmallTarget(Dataset):
    def __init__(self, length: int = 200, img_size: int = 256, num_targets: Tuple[int, int] = (1, 3)):
        self.length = length
        self.img_size = img_size
        self.num_targets = num_targets

    def __len__(self):
        return self.length

    def _gen_background(self, size: int) -> np.ndarray:
        bg = np.random.randn(size, size).astype(np.float32)
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=3)
        bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)
        bg = (bg * 200 + 30)  # low-contrast
        return bg.astype(np.uint8)

    def _draw_gaussian(self, img: np.ndarray, x: int, y: int, sigma: float, intensity: float):
        xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        g = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
        g = g / (g.max() + 1e-6)
        img += (g * intensity).astype(np.uint8)
        img[:] = np.clip(img, 0, 255)
        return img

    def __getitem__(self, idx: int):
        size = self.img_size
        img = self._gen_background(size)
        mask = np.zeros((size, size), dtype=np.uint8)
        n = random.randint(self.num_targets[0], self.num_targets[1])
        for _ in range(n):
            x = random.randint(4, size - 5)
            y = random.randint(4, size - 5)
            sigma = random.uniform(0.6, 1.5)
            intensity = random.uniform(30, 80)
            img = self._draw_gaussian(img, x, y, sigma, intensity)
            cv2.circle(mask, (x, y), max(1, int(2 * sigma)), 255, -1)
        image_t = _to_tensor(img)
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return {
            'image': image_t,
            'mask': mask_t,
            'name': f'synth_{idx:06d}.png',
        }


def create_dataloader(dataset: str, data_root: Optional[str], img_size: int, batch_size: int, workers: int,
                      split: str = 'train') -> DataLoader:
    if dataset == 'synthetic':
        ds = SyntheticIrSmallTarget(length=200 if split == 'train' else 50, img_size=img_size)
    elif dataset == 'custom':
        if data_root is None:
            raise ValueError('data_root must be provided for custom dataset')
        ds = IRTargetDataset(root=data_root, img_size=img_size, split=split, augment=(split == 'train'))
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'), num_workers=workers, pin_memory=True)