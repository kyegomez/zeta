"""
Implementatin of the sparse attention of the transformer





"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == "all":
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == "local":
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == "strided":
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError("Not yet implemented")
    b = torch.reshape(b, [1, 1, n, n])
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f"{bT_ctx}, {blocksize}"
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = torch.transpose(x, 0, 2, 1, 3)
    x = torch.reshape(x, [n, t, embd])
    return x


def split_heads(x, n):
    return torch.transpose(split_states(x, n), 0, 2, 1, 3)


def merge_heads(x):
    return merge_states(torch.transpose(x, 0, 2, 1, 3))


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)

    return torch.reshape(x, new_x_shape)


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)


def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a


def blocksparse_attention_impl(
    q,
    k,
    v,
    heads,
    attn_mode,
    local_attn_ctx=None,
    blocksize=32,
    num_verts=None,
    vertsize=None,
):
    n_ctx = q.size()[1]
    if attn_mode == "strided":
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == "strided":
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = torch.transpose(a, 0, 2, 1, 3)
        a = torch.reshape(a, [n, t, embd])
    return a


class SparseAttention(nn.Module):
    """
    Sparse Attention

    Args:
        heads: number of heads
        attn_mode: 'all', 'local', 'strided'
        local_attn_ctx: context for local attention
        blocksize: blocksize for strided attention

    Attributes:
        heads: number of heads
        attn_mode: 'all', 'local', 'strided'
        local_attn_ctx: context for local attention
        blocksize: blocksize for strided attention

    Usage:

    Example usage:
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    heads = 4
    attn_mode = "all"
    local_attn_ctx = 32
    blocksize = 32

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    model = SparseAttention(heads, attn_mode, local_attn_ctx, blocksize)
    output = model(q, k, v)
    print(output[0])


    """

    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        """Forward pass of the sparse attention"""
        return blocksparse_attention_impl(
            q, k, v, self.heads, self.attn_mode, self.local_attn_ctx
        )
