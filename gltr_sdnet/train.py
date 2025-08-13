import os
import argparse
import torch

from .utils.train_utils import set_seed, load_config, build_model, build_optimizer, train_one_epoch, validate, save_checkpoint
from .utils.datasets import create_dataloader
from .utils.loss import GLTRLoss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='gltr_sdnet/configs/default.yaml')
    ap.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'custom'])
    ap.add_argument('--data-root', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.img_size is not None:
        cfg['train']['img_size'] = args.img_size
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size
    if args.workers is not None:
        cfg['train']['workers'] = args.workers
    if args.save_dir is not None:
        cfg['logging']['save_dir'] = args.save_dir

    set_seed(cfg['train'].get('seed', 42))
    device = torch.device(args.device)

    train_loader = create_dataloader(args.dataset, args.data_root, cfg['train']['img_size'], cfg['train']['batch_size'], cfg['train']['workers'], 'train')
    val_loader = create_dataloader(args.dataset, args.data_root, cfg['train']['img_size'], cfg['train']['batch_size'], cfg['train']['workers'], 'val')

    model = build_model(cfg).to(device)
    optimizer, scheduler = build_optimizer(model, cfg)
    loss_fn = GLTRLoss(
        focal_alpha=cfg['loss'].get('focal_alpha', 0.25),
        focal_gamma=cfg['loss'].get('focal_gamma', 2.0),
        lambda_lovasz=cfg['loss'].get('lambda_lovasz', 1.0),
        lambda_boundary=cfg['loss'].get('lambda_boundary', 1.0),
        lambda_bg=cfg['loss'].get('lambda_bg', 0.5),
    )

    save_dir = cfg['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg['logging'].get('print_freq', 50))
        val_metrics = validate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        f1 = val_metrics['f1']
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f} val_f1={f1:.4f} val_miou={val_metrics['miou']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, epoch, save_dir, best=True)


if __name__ == '__main__':
    main()