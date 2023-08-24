from abc import  abstractmethod
import torch.nn as nn

class BaseAttention(nn.Module):
    @abstractmethod
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass