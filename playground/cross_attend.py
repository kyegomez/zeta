"""
Docstring for playground/cross_attend.py
"""

import torch

from zeta.nn.attention.cross_attention import CrossAttend
from zeta.structs.transformer import Encoder

encoder = Encoder(dim=512, depth=6)
model = CrossAttend(dim=512, depth=6)

nodes = torch.randn(1, 1, 512)
node_mask = torch.ones(1, 1).bool()

neighbors = torch.randn(1, 5, 512)
neighbor_mask = torch.ones(1, 5).bool()

encoded_neighbors = encoder(neighbors, mask=neighbor_mask)
model(
    nodes, context=encoded_neighbors, mask=node_mask, context_mask=neighbor_mask
)
