import torch

from yolomhaf import YOLOMHAF


def main():
    model = YOLOMHAF(num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        preds = model(x)
    for i, p in enumerate(preds):
        print(f"P{i+3}: shape={tuple(p.shape)}")


if __name__ == "__main__":
    main()