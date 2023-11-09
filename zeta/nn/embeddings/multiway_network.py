# Copyright (c) 2022 Agora
# Licensed under The MIT License [see LICENSE for details]

import copy

import torch
import torch.nn as nn


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    """
    Multiway

    Args:
        module (nn.Module): The module to apply multi-way to.
        dim (int): The dimension along which to split and concatenate the input tensor. Default is 1.

    Attributes:
        A (nn.Module): The first copy of the module.
        B (nn.Module): The second copy of the module.
        split_position (int): The position along the dimension to split the input tensor.

    Example:
        >>> module = nn.Linear(10, 10)
        >>> module = MultiwayNetwork(module)
        >>> x = torch.randn(10, 10)
        >>> y = module(x)
        >>> y.shape
        torch.Size([10, 10])
        >>> module.split_position = 5
        >>> y = module(x)
        >>> y.shape
        torch.Size([10, 20])

    """

    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MultiwayEmbedding(MultiwayNetwork):
    """
    A specialized version of the MultiwayNetwork to perform multi-way embeddings on an input tensor.

    Parameters:
    - modules (List[nn.Module]): A list containing exactly two PyTorch modules. Typically these would be embedding layers.
    - dim (int): The dimension along which to split and concatenate the input tensor. Default is 1.
    """

    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1
