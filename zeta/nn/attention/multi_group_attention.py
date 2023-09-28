from typing import Any, Optional

import torch
from torch import nn

from zeta.nn.attention.attend import Attend


class MultiGroupQueryAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads: int = None,
            softmax_scale: Optional[float] = None,
            attn_pdrop: float = 0.0,
            device: Optional[str] = None,
            kv_heads: int = None
    ):
        super(MultiGroupQueryAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.softmax_scale = softmax_scale
        
        self.attn_pdrop = attn_pdrop
        self.device = device
        self.kv_heads = kv_heads

    def forward(self):
        pass