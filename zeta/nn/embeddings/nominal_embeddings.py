# Copyright (c) 2022 Agora
# Licensed under The MIT License [see LICENSE for details]

from torch import nn
from zeta.nn.embeddings.base import BaseEmbedding

# Other embedding


class NominalEmbedding(BaseEmbedding):
    def forward(self, num_tokens: int, dim: int) -> nn.Module:
        embedding = nn.Embedding(num_tokens, dim)

        return embedding
