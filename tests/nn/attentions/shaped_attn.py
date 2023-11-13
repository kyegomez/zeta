import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta.nn.attention.shaped_attention import ShapedAttention


# Test case for initializing the ShapedAttention module
def test_shaped_attention_init():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)
    assert isinstance(shaped_attention, ShapedAttention)


# Test case for the forward pass of the ShapedAttention module
def test_shaped_attention_forward():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Create a random input tensor
    x = torch.randn(1, 32, dim)

    # Perform a forward pass
    out = shaped_attention(x)

    # Check if the output shape matches the input shape
    assert out.shape == (1, 32, dim)


# Test case for customizing the alpha, beta, and gamma parameters
def test_shaped_attention_custom_params():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Customize alpha, beta, and gamma
    shaped_attention.alpha.data = torch.ones(1, heads, 1, 1) * 0.5
    shaped_attention.beta.data = torch.ones(1, heads, 1, 1) * 0.2
    shaped_attention.gamma.data = torch.ones(1, heads, 1, 1) * 0.1

    # Create a random input tensor
    x = torch.randn(1, 32, dim)

    # Perform a forward pass
    out = shaped_attention(x)

    # Check if the output shape matches the input shape
    assert out.shape == (1, 32, dim)


# Test case for dropout rate
def test_shaped_attention_dropout():
    dim = 768
    heads = 8
    dropout = 0.5

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Create a random input tensor
    x = torch.randn(1, 32, dim)

    # Perform a forward pass
    out = shaped_attention(x)

    # Check if dropout has been applied (output should not be identical)
    assert not torch.allclose(out, x)


# Test case for the scale factor in attention calculation
def test_shaped_attention_scale_factor():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Create a random input tensor
    x = torch.randn(1, 32, dim)

    # Perform a forward pass
    out = shaped_attention(x)

    # Calculate the scale factor manually
    scale_factor = (dim // heads) ** -0.5

    # Check if the attention scores are scaled correctly
    assert torch.allclose(out, x * scale_factor)


# Test case for the case where alpha, beta, and gamma are all zeros
def test_shaped_attention_zero_params():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Set alpha, beta, and gamma to zeros
    shaped_attention.alpha.data = torch.zeros(1, heads, 1, 1)
    shaped_attention.beta.data = torch.zeros(1, heads, 1, 1)
    shaped_attention.gamma.data = torch.zeros(1, heads, 1, 1)

    # Create a random input tensor
    x = torch.randn(1, 32, dim)

    # Perform a forward pass
    out = shaped_attention(x)

    # Check if the output is identical to the input
    assert torch.allclose(out, x)


# Test case for gradient checking using torch.autograd.gradcheck
def test_shaped_attention_gradient_check():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Create a random input tensor
    x = torch.randn(1, 32, dim)
    x.requires_grad = True

    # Perform a forward pass and backward pass
    out = shaped_attention(x)
    grad_output = torch.randn_like(out)
    torch.autograd.gradcheck(shaped_attention, (x,), grad_output)


# Test case for input with zero values
def test_shaped_attention_zero_input():
    dim = 768
    heads = 8
    dropout = 0.1

    shaped_attention = ShapedAttention(dim, heads, dropout)

    # Create an input tensor with all zeros
    x = torch.zeros(1, 32, dim)

    # Perform a forward pass
    out = shaped_attention(x)

    # Check if the output is identical to the input
    assert torch.allclose(out, x)
