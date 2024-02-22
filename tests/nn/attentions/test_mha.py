import pytest
import torch

from zeta.nn.attention.multihead_attention import MultiheadAttention


def test_multiheadattention_initialization():
    args = {"layernorm_eps": 1e-5, "xpos_rel_pos": False}
    model = MultiheadAttention(args, embed_dim=512, num_heads=8)
    assert isinstance(model, MultiheadAttention)
    assert model.embed_dim == 512
    assert model.num_heads == 8
    assert model.head_dim == 64
    assert model.scaling == 1 / 8


def test_multiheadattention_forward():
    args = {"layernorm_eps": 1e-5, "xpos_rel_pos": False}
    model = MultiheadAttention(args, embed_dim=512, num_heads=8)
    query = torch.randn(1, 10, 512)
    key = torch.randn(1, 10, 512)
    value = torch.randn(1, 10, 512)
    output, attn_weights = model(query, key, value)
    assert output.shape == (1, 10, 512)
    assert attn_weights.shape == (8, 1, 10, 10)


@pytest.mark.parametrize(
    "query_len, key_len, value_len", [(0, 10, 10), (10, 0, 10), (10, 10, 0)]
)
def test_multiheadattention_forward_edge_cases(query_len, key_len, value_len):
    args = {"layernorm_eps": 1e-5, "xpos_rel_pos": False}
    model = MultiheadAttention(args, embed_dim=512, num_heads=8)
    query = torch.randn(1, query_len, 512)
    key = torch.randn(1, key_len, 512)
    value = torch.randn(1, value_len, 512)
    with pytest.raises(Exception):
        model(query, key, value)


def test_multiheadattention_forward_invalid_dimensions():
    args = {"layernorm_eps": 1e-5, "xpos_rel_pos": False}
    model = MultiheadAttention(args, embed_dim=512, num_heads=8)
    query = torch.randn(1, 10, 256)
    key = torch.randn(1, 10, 512)
    value = torch.randn(1, 10, 512)
    with pytest.raises(Exception):
        model(query, key, value)
