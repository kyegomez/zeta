from abc import ABC, abstractmethod

from torch import nn


class BaseEmbedding(ABC):
    @abstractmethod
    def forward(self, num_tokens: int, dim: int) -> nn.Module:
        # custom embedding function
        embedding = ...

        return embedding
