import argparse
import torch
import os

from .utils.train_utils import load_config, build_model
from .utils.datasets import create_dataloader
from .utils.metrics import compute_f1, compute_iou


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--config', type=str, default='gltr_sdnet/configs/default.yaml')
    ap.add_argument('--dataset', type=str, default='custom', choices=['synthetic', 'custom'])
    ap.add_argument('--data-root', type=str, default=None)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.img_size is not None:
        cfg['train']['img_size'] = args.img_size
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    loader = create_dataloader(args.dataset, args.data_root, cfg['train']['img_size'], cfg['train']['batch_size'], cfg['train']['workers'], 'val')

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
    print(f"F1={f1_sum/len(loader):.4f} mIoU={iou_sum/len(loader):.4f}")


if __name__ == '__main__':
    main()