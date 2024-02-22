import pytest
import torch

from zeta.nn.attention.multiquery_attention import MultiQueryAttention


def test_multiqueryattention_initialization():
    model = MultiQueryAttention(d_model=512, heads=8)
    assert isinstance(model, MultiQueryAttention)
    assert model.d_model == 512
    assert model.heads == 8
    assert model.head_dim == 64
    assert model.softmax_scale == 1 / 8


def test_multiqueryattention_forward():
    model = MultiQueryAttention(d_model=512, heads=8)
    x = torch.randn(1, 10, 512)
    output, attn_weights, past_key_value = model(x)
    assert output.shape == (1, 10, 512)
    assert attn_weights.shape == (1, 8, 10, 10)
    assert past_key_value is None


@pytest.mark.parametrize("x_len", [0])
def test_multiqueryattention_forward_edge_cases(x_len):
    model = MultiQueryAttention(d_model=512, heads=8)
    x = torch.randn(1, x_len, 512)
    with pytest.raises(Exception):
        model(x)


def test_multiqueryattention_forward_invalid_dimensions():
    model = MultiQueryAttention(d_model=512, heads=8)
    x = torch.randn(1, 10, 256)
    with pytest.raises(Exception):
        model(x)
