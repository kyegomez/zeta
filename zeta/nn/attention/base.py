from abc import abstractmethod
import torch.nn as nn


class BaseAttention(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass
