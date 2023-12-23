import torch
import pytest
from zeta.nn.modules.sparq_attn import SparQAttention


def test_sparq_attention_init():
    model = SparQAttention(4, 4)
    assert model.dim == 4
    assert model.heads == 4


def test_sparq_attention_forward():
    model = SparQAttention(4, 4)
    Q = torch.randn(2, 4, 10, 4)
    K = torch.randn(2, 4, 10, 4)
    V = torch.randn(2, 4, 10, 4)
    V_mean = torch.randn(2, 4, 1, 4)
    M = torch.randn(2, 4, 10, 10)
    r = 2
    k = 2
    out = model(Q, K, V, V_mean, M, r, k)
    assert out.shape == torch.Size([2, 4, 10, 4])


@pytest.mark.parametrize("r, k", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
def test_sparq_attention_forward_different_r_k(r, k):
    model = SparQAttention(4, 4)
    Q = torch.randn(2, 4, 10, 4)
    K = torch.randn(2, 4, 10, 4)
    V = torch.randn(2, 4, 10, 4)
    V_mean = torch.randn(2, 4, 1, 4)
    M = torch.randn(2, 4, 10, 10)
    out = model(Q, K, V, V_mean, M, r, k)
    assert out.shape == torch.Size([2, 4, 10, 4])


@pytest.mark.parametrize("dim, heads", [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
def test_sparq_attention_init_different_dim_heads(dim, heads):
    model = SparQAttention(dim, heads)
    assert model.dim == dim
    assert model.heads == heads


@pytest.mark.parametrize("dim, heads", [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
def test_sparq_attention_forward_different_dim_heads(dim, heads):
    model = SparQAttention(dim, heads)
    Q = torch.randn(2, heads, 10, dim)
    K = torch.randn(2, heads, 10, dim)
    V = torch.randn(2, heads, 10, dim)
    V_mean = torch.randn(2, heads, 1, dim)
    M = torch.randn(2, heads, 10, 10)
    r = 2
    k = 2
    out = model(Q, K, V, V_mean, M, r, k)
    assert out.shape == torch.Size([2, heads, 10, dim])
