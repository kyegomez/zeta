import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


# helpers
def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# residual
# normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(
            max_seq_len, device=device, dtype=self.inv_freq.dtype
        )
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# Assuming necessary imports like RotaryEmbedding, SwiGLU, etc. are present


def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (
            attn_inner_dim,
            dim_head,
            dim_head,
            (ff_inner_dim * 2),
        )

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(
            dim, sum(self.fused_dims), bias=False
        )
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.feedforward = (FeedForward(dim, dim, dropout=0.1),)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ParallelTransformerBlock(dim, dim_head, heads, ff_mult),
                        FeedForward(dim, dim, dropout=0.1),
                    ]
                )
            )

    def forward(self, x):
        for block, ff in self.layers:
            x = block(x) + x
            x = ff(x) + x
        return x


# classes


class SimpleTransformer(nn.Module):
    """

    Simple Transformer

    Args:
        dim (int): The feature dimension.
        depth (int): The depth of the transformer.
        num_tokens (int): The number of tokens.
        dim_head (int): The dimension of the heads.
        heads (int): The number of heads.
        ff_mult (int): The feedforward multiplier.


    Attributes:
        emb (nn.Embedding): The embedding layer.
        transformer (Transformer): The transformer.
        to_logits (nn.Sequential): The logits layer.

    Example:
        >>> module = SimpleTransformer(512, 6, 20000)
        >>> x = torch.randn(2, 1024).long()
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 1024, 20000])


    """

    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, ff_mult)

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        """Forward method implementation."""
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)


# autoregressive wrapper for generation
class AutoregressiveWrapper(nn.Module):
    """
    Autoregressive Wrapper

    Args:
        net (nn.Module): The module.
        max_seq_len (int): The maximum sequence length.
        pad_value (int): The pad value.

    Example:
        >>> module = AutoregressiveWrapper(nn.Linear(10, 10))
        >>> x = torch.randn(2, 1024).long()
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 1024, 10])

    """

    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_thres=0.9,
        **kwargs,
    ):
        """
        Args:

            start_tokens (torch.Tensor): The start tokens.
            seq_len (int): The sequence length.
            eos_token (int): The end of sequence token.
            temperature (float): The temperature.
            filter_thres (float): The filter threshold.
            kwargs (dict): The keyword arguments.

        Returns:
            torch.Tensor: The generated tokens.

        Example:
            >>> module = AutoregressiveWrapper(nn.Linear(10, 10))
            >>> x = torch.randn(2, 1024).long()
            >>> y = module(x)
            >>> y.shape
            torch.Size([2, 1024, 10])


        """
        # einops conflicts with ruff, so noqa on next line
        b, t, device = *start_tokens.shape, start_tokens.device  # noqa F841

        out = start_tokens

        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_token = out == eos_token

                if is_eos_token.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]
        return out

    def forward(self, x, **kwargs):
        """Forward method implementation."""
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        return F.cross_entropy(rearrange(logits, "b c n -> b n c"), x_labels)
