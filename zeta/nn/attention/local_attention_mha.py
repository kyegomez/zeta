import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from zeta.nn.attention.local_attention import LocalAttention
from zeta.utils.main import default, exists, l2norm


class LocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        prenorm=False,
        qk_rmsnorm=False,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=causal,
            autopad=True,
            scale=(qk_scale if qk_rmsnorm else None),
            exact_windowsize=default(exact_windowsize, True),
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs,
        )

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, attn_bias=None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask=mask, attn_bias=attn_bias)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
