import torch
from einops.layers.torch import Rearrange
from torch import nn

from zeta.nn.embeddings.patch_embedding import PatchEmbeddings


# Test case for default initialization
def test_default_init():
    dim_in = 3
    dim_out = 4
    seq_len = 5
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    assert module.dim_in == dim_in
    assert module.dim_out == dim_out
    assert module.seq_len == seq_len
    assert isinstance(module.embedding, nn.Sequential)


# Test case for forward pass
def test_forward_pass():
    dim_in = 3
    dim_out = 4
    seq_len = 5
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    x = torch.randn(2, dim_in, seq_len, seq_len)
    y = module(x)
    assert y.shape == (2, dim_out, seq_len)


# Test case for patch embedding size
def test_patch_embedding_size():
    dim_in = 3
    dim_out = 4
    seq_len = 5
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    x = torch.randn(2, dim_in, seq_len, seq_len)
    y = module(x)
    assert y.shape == (2, dim_out, seq_len)


# Test case for the presence of specific layers in the sequential embedding
def test_embedding_layers():
    dim_in = 3
    dim_out = 4
    seq_len = 5
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    assert isinstance(module.embedding[0], Rearrange)
    assert isinstance(module.embedding[1], nn.LayerNorm)
    assert isinstance(module.embedding[2], nn.Linear)
    assert isinstance(module.embedding[3], nn.LayerNorm)


# Test case for different input dimensions
def test_different_input_dimensions():
    dim_in = 3
    dim_out = 4
    seq_len = 5
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    x = torch.randn(2, dim_in, seq_len, seq_len)
    y = module(x)
    assert y.shape == (2, dim_out, seq_len)


# Test case for large input dimensions
def test_large_input_dimensions():
    dim_in = 256
    dim_out = 512
    seq_len = 16
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    x = torch.randn(2, dim_in, seq_len, seq_len)
    y = module(x)
    assert y.shape == (2, dim_out, seq_len)


# Test case for forward pass with a single batch and sequence length
def test_forward_pass_single_batch_sequence_length():
    dim_in = 3
    dim_out = 4
    seq_len = 5
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    x = torch.randn(1, dim_in, seq_len, seq_len)
    y = module(x)
    assert y.shape == (1, dim_out, seq_len)


# Test case for forward pass with no sequence length
def test_forward_pass_no_sequence_length():
    dim_in = 3
    dim_out = 4
    seq_len = 0
    module = PatchEmbeddings(dim_in, dim_out, seq_len)
    x = torch.randn(2, dim_in, 5, 5)
    y = module(x)
    assert y.shape == (2, dim_out, 0)
