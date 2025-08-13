import torch

def threshold(pred: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (pred > thr).float()


def compute_iou(pred_logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    pred = threshold(torch.sigmoid(pred_logits), thr)
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = (pred + targets - pred * targets).sum(dim=(1, 2, 3)) + 1e-6
    iou = (inter / union).mean().item()
    return iou


def compute_f1(pred_logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    pred = threshold(torch.sigmoid(pred_logits), thr)
    tp = (pred * targets).sum(dim=(1, 2, 3))
    fp = (pred * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * targets).sum(dim=(1, 2, 3))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2 * precision * recall / (precision + recall + 1e-6)).mean().item()
    return f1