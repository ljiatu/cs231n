import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, _assert_no_grad


class SoftArgmaxLoss(_Loss):
    """
    Implements soft argmax loss.

    Given an array of scores as inputs with each score assigned to a class, first computes
    the expected class (which is a float), and then compute the MSELoss with the target.
    """
    def __init__(self, size_average=True, reduce=True):
        super(SoftArgmaxLoss, self).__init__(size_average, reduce)

    def forward(self, x, target):
        _assert_no_grad(target)
        num_classes = x.size(1)
        expected_class = (F.softmax(x, dim=1) * torch.arange(end=num_classes, requires_grad=True).cuda()).sum(dim=1)
        return F.mse_loss(expected_class, target, size_average=self.size_average, reduce=self.reduce)
