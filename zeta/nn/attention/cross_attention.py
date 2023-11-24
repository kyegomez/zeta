import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from zeta.nn.modules.layernorm import LayerNorm, l2norm
from zeta.utils.main import exists


class CrossAttention(nn.Module):
    """
    Cross-Attention module.

    Args:
        dim (int): The dimension of the input tensor.
        context_dim (int, optional): The dimension of the context tensor. Default is None.
        dim_head (int, optional): The dimension of each attention head. Default is 64.
        heads (int, optional): The number of attention heads. Default is 8.
        dropout (float, optional): The dropout rate. Default is 0.
        norm_context (bool, optional): Whether to apply layer normalization to the context tensor. Default is False.
        cosine_sim (bool, optional): Whether to use cosine similarity for attention scores. Default is False.
        cosine_sim_scale (int, optional): The scale factor for cosine similarity. Default is 16.

    Attributes:
        cosine_sim (bool): Whether to use cosine similarity for attention scores.
        scale (float): The scale factor for attention scores.
        heads (int): The number of attention heads.
        norm (LayerNorm): The layer normalization module for the input tensor.
        norm_context (LayerNorm or nn.Identity): The layer normalization module or identity function for the context tensor.
        dropout (nn.Dropout): The dropout module.
        null_kv (nn.Parameter): The learnable null key-value parameter.
        to_q (nn.Linear): The linear transformation module for the input tensor.
        to_k (nn.Linear): The linear transformation module for the context tensor.
        to_out (nn.Sequential): The sequential module for the output tensor.

    # Usage
    ```
    import torch

    # Create an instance of CrossAttention
    cross_attention = CrossAttention(dim=512, context_dim=256)

    # Create random input and context tensors
    x = torch.randn(32, 10, 512)
    context = torch.randn(32, 20, 256)

    # Apply cross-attention
    output = cross_attention(x, context)

    # Print the output tensor
    print(output)
    ```


    """

    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        dropout=0.0,
        norm_context=False,
        cosine_sim=False,
        cosine_sim_scale=16,
    ):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.norm_context = (
            LayerNorm(context_dim) if norm_context else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(inner_dim))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim**2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        """
        Forward pass of the Cross-Attention module.

        Args:
            x (torch.Tensor): The input tensor.
            context (torch.Tensor): The context tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Default is None.

        Returns:
            torch.Tensor: The output tensor.

        """
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange("b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        # add null key value for classifier free guidance in propr
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
            mask = rearrange(mask, "b n -> b 1 1 j")
            sim = sim.msked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
