import torch
import pytest
from zeta.nn.modules.adaptive_layernorm import AdaptiveLayerNorm


def test_adaptive_layer_norm_init():
    model = AdaptiveLayerNorm(4)
    assert model.num_features == 4
    assert model.eps == 1e-5
    assert isinstance(model.gamma, torch.nn.Parameter)
    assert isinstance(model.beta, torch.nn.Parameter)


def test_adaptive_layer_norm_init_invalid_num_features():
    with pytest.raises(ValueError):
        AdaptiveLayerNorm(-1)


def test_adaptive_layer_norm_init_invalid_eps():
    with pytest.raises(ValueError):
        AdaptiveLayerNorm(4, -1)


def test_adaptive_layer_norm_forward():
    model = AdaptiveLayerNorm(4)
    x = torch.randn(2, 4, 10)
    out = model(x)
    assert out.shape == torch.Size([2, 4, 10])


def test_adaptive_layer_norm_forward_zero():
    model = AdaptiveLayerNorm(4)
    x = torch.zeros(2, 4, 10)
    out = model(x)
    assert torch.all(out == 0)


def test_adaptive_layer_norm_forward_one():
    model = AdaptiveLayerNorm(4)
    x = torch.ones(2, 4, 10)
    out = model(x)
    assert torch.all(out == model.beta)
