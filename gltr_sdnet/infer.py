import argparse
import os
import glob
import torch
import cv2
import numpy as np

from .utils.train_utils import load_config, build_model


def load_image(path: str, img_size: int) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    return t


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--config', type=str, default='gltr_sdnet/configs/default.yaml')
    ap.add_argument('--source', type=str, required=True)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--save-dir', type=str, default='runs/infer')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--thr', type=float, default=0.5)
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
    model.eval()

    paths = []
    if os.path.isdir(args.source):
        paths = sorted(glob.glob(os.path.join(args.source, '*')))
    else:
        paths = [args.source]
    os.makedirs(args.save_dir, exist_ok=True)

    for p in paths:
        try:
            img_t = load_image(p, cfg['train']['img_size']).to(device)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.sigmoid(out['logits'])[0, 0].cpu().numpy()
            mask = (prob > args.thr).astype(np.uint8) * 255
            vis = (prob * 255).astype(np.uint8)
            base = os.path.basename(p)
            cv2.imwrite(os.path.join(args.save_dir, base.replace('.', '_prob.')), vis)
            cv2.imwrite(os.path.join(args.save_dir, base.replace('.', '_mask.')), mask)
        except Exception as e:
            print(f"Failed on {p}: {e}")


if __name__ == '__main__':
    main()