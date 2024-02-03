import pytest
import torch

from zeta.nn.attention.cross_attention import CrossAttention


@pytest.fixture
def cross_attention():
    return CrossAttention(dim=512, context_dim=256, dim_head=64, heads=8)


def test_cross_attention_initialization(cross_attention):
    assert isinstance(cross_attention, CrossAttention)
    assert cross_attention.cosine_sim is False
    assert cross_attention.scale == 0.125
    assert cross_attention.heads == 8


def test_cross_attention_forward(cross_attention):
    # Prepare the test input
    x = torch.rand(1, 10, 512)
    context = torch.rand(1, 5, 256)

    # Try normal forward pass
    output = cross_attention(x, context)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10, 512)


def test_cross_attention_forward_with_mask(cross_attention):
    # Prepare the test input
    x = torch.rand(1, 10, 512)
    context = torch.rand(1, 5, 256)
    mask = torch.tensor([[True, True, True, False, False]])

    # Try forward pass with mask
    output = cross_attention(x, context, mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10, 512)


def test_cross_attention_forward_with_cosine_similarity(cross_attention):
    # Prepare the test input
    x = torch.rand(1, 10, 512)
    context = torch.rand(1, 5, 256)
    cross_attention.cosine_sim = True

    # Try forward pass with cosine similarity
    output = cross_attention(x, context)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10, 512)


def test_cross_attention_forward_with_cosine_similarity_and_mask(
    cross_attention,
):
    # Prepare the test input
    x = torch.rand(1, 10, 512)
    context = torch.rand(1, 5, 256)
    mask = torch.tensor([[True, True, True, False, False]])
    cross_attention.cosine_sim = True

    # Try forward pass with cosine similarity and mask
    output = cross_attention(x, context, mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10, 512)


def test_cross_attention_forward_with_null_key_value(cross_attention):
    # Prepare the test input
    x = torch.rand(1, 10, 512)
    context = torch.rand(1, 5, 256)
    cross_attention.null_kv = torch.tensor([[0.5, 0.5]])

    # Try forward pass with null key/value
    output = cross_attention(x, context)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10, 512)
