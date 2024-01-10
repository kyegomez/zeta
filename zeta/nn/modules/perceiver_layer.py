from typing import Optional

import torch
from torch import Tensor, nn

from zeta.nn.attention.cross_attention import CrossAttention
from zeta.nn.attention.multiquery_attention import MultiQueryAttention


class PerceiverLayer(nn.Module):
    """
    Perceiver Layer, this layer has a self attn that takes in q then ->
    sends the output into the q of the cross attention where the cross attn
    takes in k and v. The output of the cross attn is then sent into a
    feed forward layer.


    Args:
        dim: dimension of the input tensor
        heads: number of heads
        depth: number of layers
        dim_head: dimension of each head
        dropout: dropout rate
        ff_dropout: feed forward dropout rate
        ff_mult: feed forward multiplier

    Examples::
        >>> q = torch.randn(1, 32, 512)
        >>> k = torch.randn(1, 32, 512)
        >>> v = torch.randn(1, 32, 512)
        >>> layer = PerceiverLayer(512, 8, 6, 64)
        >>> print(layer(q, k, v).shape)
        torch.Size([1, 32, 512])

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_dropout = ff_dropout
        self.ff_mult = ff_mult

        # Initialize layers for MultiQueryAttention, CrossAttention, and Feed Forward
        self.self_attn = MultiQueryAttention(
            dim,
            heads,
            # qk_ln=True,
        )

        # CrossAttention initialization
        self.cross_attn = CrossAttention(
            dim,
            context_dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
        )

        # Feed Forward initialization
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(ff_dropout),
        )

        # Projection layers for x to -> q, k, v
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        """
        Args:
            q: query tensor
            k: key tensor
            v: value tensor
            mask: mask tensor

        Shape:
            q: (batch_size, seq_len_q, dim)
            k: (batch_size, seq_len_k, dim)
            v: (batch_size, seq_len_v, dim)
            mask: (batch_size, seq_len_q, seq_len_k)
        """
        q, _, _ = self.self_attn(q)

        # Concatenate k and v
        kv = torch.concat((k, v), dim=1)

        # Send q, k, v into cross attention with q as the context
        x = self.cross_attn(kv, q)

        # Apply feed forward layer to output of cross attention
        x = self.ffn(x)

        # Return output
        return x
