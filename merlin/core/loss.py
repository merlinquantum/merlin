'''
Specialized loss functions for QML
'''
import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor

class NKernelAlignment(_Loss):
    r"""
    Negative kernel-target alignment loss function for quantum kernel training.

    Within quantum kernel alignment, the goal is to maximize the
    alignment between the quantum kernel matrix and the ideal
    target matrix given by :math:`K^{*} = y y^T`, where
    :math:`y \in \{-1, +1\}` are the target labels.

    The negative kernel alignment loss is given as:

    .. math::

        \text{NKA}(K, K^{*}) =
        -\frac{\operatorname{Tr}(K K^{*})}{
        \sqrt{\operatorname{Tr}(K^2)\operatorname{Tr}(K^{*2})}}
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.dim() != 2:
            raise ValueError(
                'Input must be a 2D tensor representing the kernel matrix.')

        if torch.any((target != 1) & (target != -1)):
            raise ValueError(
                'Negative kernel alignment requires binary target values +1, -1.')

        if target.dim() == 1:
            # Make the target the ideal Kernel matrix
            target = target.unsqueeze(1) @ target.unsqueeze(0)

        numerator = torch.sum(input * target)
        denominator = torch.linalg.norm(input) * torch.linalg.norm(target)
        return -numerator / denominator
