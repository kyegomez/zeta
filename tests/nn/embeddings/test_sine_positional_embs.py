import pytest
import torch

from zeta.nn.embeddings.sine_positional import SinePositionalEmbedding


# Test case for default initialization
def test_default_init():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model)
    assert module.dim_model == dim_model
    assert module.x_scale == 1.0
    assert module.alpha.item() == 1.0
    assert module.dropout.p == 0.0


# Test case for initializing with scale=True
def test_scale_parameter():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model, scale=True)
    assert module.x_scale == pytest.approx(22.62741699)  # sqrt(512)


# Test case for initializing with alpha=True
def test_alpha_parameter():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model, alpha=True)
    assert module.alpha.requires_grad


# Test case for initializing with dropout
def test_dropout_parameter():
    dim_model = 512
    dropout = 0.2
    module = SinePositionalEmbedding(dim_model, dropout=dropout)
    assert module.dropout.p == dropout


# Test case for forward pass with 2D input
def test_forward_pass_2d_input():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model)
    x = torch.randn(1, 4000, dim_model)
    output = module(x)
    assert output.shape == (1, 4000, dim_model)


# Test case for forward pass with 3D input
def test_forward_pass_3d_input():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model)
    x = torch.randn(1, 4000, 50, dim_model)
    output = module(x)
    assert output.shape == (1, 4000, 50, dim_model)


# Test case for forward pass with scale=True
def test_forward_pass_with_scale():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model, scale=True)
    x = torch.randn(1, 4000, dim_model)
    output = module(x)
    assert output.max().item() <= 23.0  # Scaled by sqrt(dim_model)


# Test case for extending positional encodings
def test_extend_pe():
    dim_model = 512
    module = SinePositionalEmbedding(dim_model)
    x = torch.randn(1, 4000, dim_model)
    module.extend_pe(x)
    assert module.pe.shape == (1, 4000, dim_model)


# Test case for initializing with negative dimension
def test_negative_dimension():
    dim_model = -512
    with pytest.raises(ValueError):
        SinePositionalEmbedding(dim_model)


# Test case for initializing with alpha=True and dropout > 0
def test_alpha_and_dropout():
    dim_model = 512
    with pytest.raises(ValueError):
        SinePositionalEmbedding(dim_model, alpha=True, dropout=0.2)
