import torch
from torch.nn import functional as F


def check_result(scores, y) -> (int, int):
    num_classes = scores.size(1)
    expected_classes = (
        (F.softmax(scores, dim=1) * torch.arange(end=num_classes).cuda()).sum(dim=1).round().type(torch.cuda.LongTensor)
    )
    num_correct = (expected_classes == y.type(torch.cuda.LongTensor)).sum()
    num_samples = scores.size(0)
    return num_correct, num_samples
