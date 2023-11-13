import pytest
import torch
from torch import nn
from zeta.nn.attention.cross_attention import CrossAttention

# Create an instance of CrossAttention for testing
cross_attention = CrossAttention(dim=512, context_dim=256, heads=4)


# Test the forward pass of CrossAttention
def test_cross_attention_forward():
    x = torch.randn(32, 10, 512)
    context = torch.randn(32, 20, 256)
    output = cross_attention(x, context)
    assert output.shape == (32, 10, 512)


# Test forward pass with cosine similarity
def test_cross_attention_cosine_similarity():
    cosine_attention = CrossAttention(
        dim=512, context_dim=256, heads=4, cosine_sim=True
    )
    x = torch.randn(32, 10, 512)
    context = torch.randn(32, 20, 256)
    output = cosine_attention(x, context)
    assert output.shape == (32, 10, 512)


# Test forward pass with mask
def test_cross_attention_with_mask():
    x = torch.randn(32, 10, 512)
    context = torch.randn(32, 20, 256)
    mask = torch.randint(0, 2, size=(32, 10), dtype=torch.bool)
    output = cross_attention(x, context, mask=mask)
    assert output.shape == (32, 10, 512)


# Test forward pass with layer normalization
def test_cross_attention_with_layer_norm():
    layer_norm_attention = CrossAttention(
        dim=512, context_dim=256, heads=4, norm_context=True
    )
    x = torch.randn(32, 10, 512)
    context = torch.randn(32, 20, 256)
    output = layer_norm_attention(x, context)
    assert output.shape == (32, 10, 512)


# Test forward pass with dropout
def test_cross_attention_with_dropout():
    dropout_attention = CrossAttention(dim=512, context_dim=256, heads=4, dropout=0.1)
    x = torch.randn(32, 10, 512)
    context = torch.randn(32, 20, 256)
    output = dropout_attention(x, context)
    assert output.shape == (32, 10, 512)
