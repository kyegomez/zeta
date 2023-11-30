import pytest
import torch
from torch import nn
from zeta.nn.attention import SparseAttention


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
        "zeta.nn.attention.sparse_attention.blocksparse_attention_impl",
        mock_blocksparse_attention_impl,
    )
    q, k, v = input_tensors
    output = sparse_attention(q, k, v)
    assert torch.allclose(output, q + k + v)


@pytest.mark.parametrize("attn_mode", ["all", "local", "strided"])
def test_attn_modes(sparse_attention, input_tensors, attn_mode, monkeypatch):
    monkeypatch.setattr(
        "zeta.nn.attention.sparse_attention.blocksparse_attention_impl",
        mock_blocksparse_attention_impl,
    )
    sparse_attention.attn_mode = attn_mode
    q, k, v = input_tensors
    output = sparse_attention(q, k, v)
    assert torch.allclose(output, q + k + v)


# Helper function to check if a tensor is sparse (contains zeros)
def is_sparse(tensor):
    return (tensor == 0).all()


# Test the forward pass of SparseAttention
def test_sparse_attention_forward():
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

    output = sparse_attention(q, k, v)
    assert output.shape == (n_batch, n_ctx, n_embd)


# Test SparseAttention with different head counts
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_sparse_attention_with_different_heads(heads):
    attn_mode = "all"
    local_attn_ctx = 32
    blocksize = 32

    sparse_attention = SparseAttention(
        heads=heads,
        attn_mode=attn_mode,
        local_attn_ctx=local_attn_ctx,
        blocksize=blocksize,
    )

    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = sparse_attention(q, k, v)
    assert output.shape == (n_batch, n_ctx, n_embd)


# Test SparseAttention with different attention modes
@pytest.mark.parametrize("attn_mode", ["all", "local", "strided"])
def test_sparse_attention_with_different_modes(attn_mode):
    heads = 4
    local_attn_ctx = 32
    blocksize = 32

    sparse_attention = SparseAttention(
        heads=heads,
        attn_mode=attn_mode,
        local_attn_ctx=local_attn_ctx,
        blocksize=blocksize,
    )

    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = sparse_attention(q, k, v)
    assert output.shape == (n_batch, n_ctx, n_embd)


# Test SparseAttention with local attention context
def test_sparse_attention_with_local_context():
    heads = 4
    attn_mode = "local"
    local_attn_ctx = 64
    blocksize = 32

    sparse_attention = SparseAttention(
        heads=heads,
        attn_mode=attn_mode,
        local_attn_ctx=local_attn_ctx,
        blocksize=blocksize,
    )

    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = sparse_attention(q, k, v)
    assert output.shape == (n_batch, n_ctx, n_embd)


# Test SparseAttention with blocksize for strided attention
def test_sparse_attention_with_blocksize():
    heads = 4
    attn_mode = "strided"
    local_attn_ctx = 32
    blocksize = 64

    sparse_attention = SparseAttention(
        heads=heads,
        attn_mode=attn_mode,
        local_attn_ctx=local_attn_ctx,
        blocksize=blocksize,
    )

    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = sparse_attention(q, k, v)
    assert output.shape == (n_batch, n_ctx, n_embd)


# Test if the output of SparseAttention is sparse when using 'all' attention mode
def test_sparse_attention_output_sparse():
    heads = 4
    attn_mode = "all"
    local_attn_ctx = 32
    blocksize = 32

    sparse_attention = SparseAttention(
        heads=heads,
        attn_mode=attn_mode,
        local_attn_ctx=local_attn_ctx,
        blocksize=blocksize,
    )

    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.zeros(n_batch, n_ctx, n_embd)  # Create a tensor with all zeros
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = sparse_attention(q, k, v)
    assert is_sparse(output)  # Check if the output is sparse


# Test if the output of SparseAttention is not sparse when using 'local' attention mode
def test_sparse_attention_output_not_sparse():
    heads = 4
    attn_mode = "local"
    local_attn_ctx = 32
    blocksize = 32

    sparse_attention = SparseAttention(
        heads=heads,
        attn_mode=attn_mode,
        local_attn_ctx=local_attn_ctx,
        blocksize=blocksize,
    )

    n_batch = 4
    n_ctx = 1024
    n_embd = 256

    q = torch.zeros(n_batch, n_ctx, n_embd)  # Create a tensor with all zeros
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    output = sparse_attention(q, k, v)
    assert not is_sparse(output)  # Check if the output is not sparse
