# Copyright (c) 2022 Agora
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @abstractmethod
    def forward(self, num_tokens: int, dim: int) -> nn.Module:
        # custom embedding function
        embedding = ...

        return embedding


# Other embedding


class Embedding(BaseEmbedding):
    def forward(self, num_tokens: int, dim: int) -> nn.Module:
        embedding = nn.Embedding(num_tokens, dim)

        return embedding


class TextEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()
