import torch
from gltr_sdnet.models.model import GLTRSDNet
from gltr_sdnet.utils.datasets import SyntheticIrSmallTarget
from gltr_sdnet.utils.loss import GLTRLoss
from torch.utils.data import DataLoader


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GLTRSDNet().to(device)
    ds = SyntheticIrSmallTarget(length=4, img_size=128)
    dl = DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    images = batch['image'].to(device)
    out = model(images)
    assert 'logits' in out and out['logits'].shape[-2:] == images.shape[-2:]
    # tiny train step
    loss_fn = GLTRLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        out = model(images)
        loss_dict = loss_fn(out, {**batch, 'image': images, 'mask': batch['mask'].to(device)})
        opt.zero_grad()
        loss_dict['loss'].backward()
        opt.step()
    print('Smoke test passed')


if __name__ == '__main__':
    run()