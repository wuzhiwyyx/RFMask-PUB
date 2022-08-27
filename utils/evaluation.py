import torch
from torchvision.ops.boxes import box_area

def accuracy(pred, target):
    """
    Args:
        pred: Tensor of size `B x C`
        target: Tensor of size `B`
    """
    return (pred.argmax(dim=1) == target).float().mean()

def box_iou(boxes1, boxes2, eps=1e-7):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    return iou, union
