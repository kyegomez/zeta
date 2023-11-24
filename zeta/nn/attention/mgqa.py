from typing import Tuple

import torch
from torch import nn

from zeta.nn.attention.attend import Attend
from zeta.nn.modules.cache import CacheView


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# mgqa
class MGQA(nn.Module):
    """
    Multi-Headed Generalized Query Attention

    Args:
        dim (int): Input dimension
        n_layers (int): Number of layers
        head_dim (int): Head dimension
        hidden_dim (int): Hidden dimension
        n_heads (int): Number of heads
        n_kv_heads (int): Number of key/value heads
        sliding_window (int): Sliding window size
        norm_eps (float): Epsilon for layer norm
        vocab_size (int): Vocabulary size
        attn_dropout (float): Dropout probability
        max_batch_size (int): Maximum batch size
        flash (bool): Use FlashAttention

    Usage:
    >>> model = MGQA(768, 12, 64, 2048, 8, 8, 512, 1e-6, 32000, 0.1, 0, False)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape


    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        head_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        sliding_window: int,
        norm_eps: float,
        vocab_size: int,
        attn_dropout: float = 0.0,  # moved to the end
        max_batch_size: int = 0,  # default argument
        flash: bool = False,  # non-default argument
    ):
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.sliding_window = sliding_window
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.attn_dropout = attn_dropout
        self.flash = flash

        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            self.n_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn = Attend(
            dropout=self.attn_dropout,
            causal=True,
            flash=self.flash,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: CacheView,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor
            freqs_cis (torch.Tensor): Precomputed frequencies
            cache (CacheView): Cache view

        Example:
        >>> model = MGQA(768, 12, 64, 2048, 8, 8, 512, 1e-6, 32000, 0.1, 0, False)
        >>> x = torch.randn(1, 768)
        >>> freqs_cis = torch.randn(1, 768)
        >>> cache = CacheView(1, 512, 8, 8, 64)
        >>> model(x, freqs_cis, cache).shape


        """
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)

        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)

        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(
            xq,
            xk,
            freqs_cis=freqs_cis,
        )

        if cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.keys, cache.values

            key = key.view(
                seqlen_sum * cache.sliding_window,
                self.n_kv_heads,
                self.head_dim,
            )

            val = val.view(
                seqlen_sum * cache.sliding_window,
                self.n_kv_heads,
                self.head_dim,
            )

        # repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # attention
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = self.attn(xq, key, val, self.scale)

        return self.wo(output.view_as(x))
