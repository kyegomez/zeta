from abc import abstractmethod
import torch.nn as nn


class BaseBias(nn.Module):
    @abstractmethod
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads

    @abstractmethod
    def forward(self):
        pass
