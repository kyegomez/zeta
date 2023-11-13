import pytest
import torch
from torch import nn

from zeta.nn.embeddings.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    exists,
    rotate_half,
)


# Test case for default initialization
def test_default_init():
    dim = 512
    module = RotaryEmbedding(dim)
    assert module.dim == dim
    assert module.use_xpos is False
    assert module.interpolation_factor == 1.0
    assert module.base == 10000
    assert module.base_rescale_factor == 1.0
    assert module.inv_freq.shape == (dim // 2,)
    assert module.scale is None


# Test case for initializing with use_xpos=True
def test_use_xpos_parameter():
    dim = 512
    module = RotaryEmbedding(dim, use_xpos=True)
    assert module.use_xpos is True
    assert module.scale_base == 512
    assert module.scale.shape == (dim // 2,)


# Test case for initializing with interpolation_factor
def test_interpolation_factor_parameter():
    dim = 512
    interpolation_factor = 2.0
    module = RotaryEmbedding(dim, interpolation_factor=interpolation_factor)
    assert module.interpolation_factor == interpolation_factor


# Test case for initializing with base_rescale_factor
def test_base_rescale_factor_parameter():
    dim = 512
    base_rescale_factor = 2.0
    module = RotaryEmbedding(dim, base_rescale_factor=base_rescale_factor)
    assert module.base_rescale_factor == base_rescale_factor


# Test case for forward pass without use_xpos
def test_forward_pass_without_use_xpos():
    dim = 512
    module = RotaryEmbedding(dim)
    seq_len = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freqs, scale = module(seq_len, device)
    assert freqs.shape == (seq_len, dim)
    assert scale == 1.0


# Test case for forward pass with use_xpos=True
def test_forward_pass_with_use_xpos():
    dim = 512
    module = RotaryEmbedding(dim, use_xpos=True)
    seq_len = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freqs, scale = module(seq_len, device)
    assert freqs.shape == (seq_len, dim)
    assert scale.shape == (seq_len, dim // 2)


# Test case for exists function
def test_exists_function():
    val = None
    assert exists(val) is False
    val = 0
    assert exists(val) is True
    val = [1, 2, 3]
    assert exists(val) is True


# Test case for rotate_half function
def test_rotate_half_function():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = rotate_half(x)
    expected = torch.tensor([-2.0, 1.0, -4.0, 3.0])
    assert torch.allclose(rotated, expected)


# Test case for apply_rotary_pos_emb function
def test_apply_rotary_pos_emb_function():
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    freqs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    scale = 2.0
    result = apply_rotary_pos_emb(t, freqs, scale)
    expected = torch.tensor([[0.0, 4.0], [1.0, 11.0], [4.0, 30.0], [11.0, 64.0]])
    assert torch.allclose(result, expected)


# Test case for applying rotary positional embedding without scale
def test_apply_rotary_pos_emb_without_scale():
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    freqs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    result = apply_rotary_pos_emb(t, freqs)
    expected = torch.tensor([[0.0, 2.0], [1.0, 10.0], [4.0, 24.0], [11.0, 48.0]])
    assert torch.allclose(result, expected)
