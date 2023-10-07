from torch import nn
from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @abstractmethod
    def forward(self, num_tokens: int, dim: int) -> nn.Module:
        # custom embedding function
        embedding = ...

        return embedding
