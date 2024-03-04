import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn import LayerNorm

from zeta.utils.main import default, exists, l2norm


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        dropout=0.1,
        norm_context=False,
        cosine_sim=False,
        cosine_sim_scale=16,
    ):
        """
        CrossAttention module performs cross-attention mechanism between input tensor `x` and context tensor `context`.

        Args:
            dim (int): The dimension of the input tensor `x`.
            context_dim (int, optional): The dimension of the context tensor `context`. If not provided, it defaults to `dim`.
            dim_head (int, optional): The dimension of each head in the multi-head attention. Defaults to 64.
            heads (int, optional): The number of attention heads. Defaults to 8.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            norm_context (bool, optional): Whether to apply layer normalization to the context tensor. Defaults to False.
            cosine_sim (bool, optional): Whether to use cosine similarity for attention calculation. Defaults to False.
            cosine_sim_scale (int, optional): The scale factor for cosine similarity. Defaults to 16.
        """
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = (
            LayerNorm(context_dim) if norm_context else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, dim).
            context (torch.Tensor): The context tensor of shape (batch_size, context_length, context_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, dim).
        """
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (
            self.to_q(x),
            *self.to_kv(context).chunk(2, dim=-1),
        )

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, "d -> b h 1 d", h=self.heads, b=b),
            self.null_kv.unbind(dim=-2),
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
