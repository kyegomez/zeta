from abc import ABC, abstractmethod
import torch.nn as nn

class BaseAttention(nn.Module):
    @abstractmethod
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass