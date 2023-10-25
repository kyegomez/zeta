import pytest
import torch
from torch import nn
from zeta.nn.attention import SparseAttention, blocksparse_attention_impl


# Mocking the blocksparse_attention_impl function
def mock_blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx):
    return q + k + v


@pytest.fixture
def sparse_attention():
    return SparseAttention(4, "all", 32, 32)


@pytest.fixture
def input_tensors():
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)
    return q, k, v


def test_init(sparse_attention):
    assert isinstance(sparse_attention, nn.Module)
    assert sparse_attention.heads == 4
    assert sparse_attention.attn_mode == "all"
    assert sparse_attention.local_attn_ctx == 32
    assert sparse_attention.blocksize == 32


def test_forward(sparse_attention, input_tensors, monkeypatch):
    monkeypatch.setattr(
        "your_module.blocksparse_attention_impl", mock_blocksparse_attention_impl
    )
    q, k, v = input_tensors
    output = sparse_attention(q, k, v)
    assert torch.allclose(output, q + k + v)


@pytest.mark.parametrize("attn_mode", ["all", "local", "strided"])
def test_attn_modes(sparse_attention, input_tensors, attn_mode, monkeypatch):
    monkeypatch.setattr(
        "your_module.blocksparse_attention_impl", mock_blocksparse_attention_impl
    )
    sparse_attention.attn_mode = attn_mode
    q, k, v = input_tensors
    output = sparse_attention(q, k, v)
    assert torch.allclose(output, q + k + v)
