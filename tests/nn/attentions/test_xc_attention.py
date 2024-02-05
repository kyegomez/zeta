""" Test cases for the XCAttention class. """
import torch
import pytest
from torch import nn

from zeta.nn.attention.xc_attention import XCAttention


@pytest.fixture
def xc_attention_model():
    """ Fixture to create an instance of the XCAttention class. """
    model = XCAttention(dim=256, cond_dim=64, heads=8, dropout=0.1)
    return model


def test_xc_attention_initialization(xc_attention_model):
    """ Test case to check if XCAttention initializes correctly. """
    assert isinstance(xc_attention_model, XCAttention)
    assert isinstance(xc_attention_model.norm, nn.LayerNorm)
    assert isinstance(xc_attention_model.to_qkv, nn.Sequential)


def test_xc_attention_forward_pass(xc_attention_model):
    """ Test case to check if XCAttention handles forward pass correctly. """
    x = torch.randn(1, 256, 16, 16)
    cond = torch.randn(1, 64)

    output = xc_attention_model(x, cond)

    assert isinstance(output, torch.Tensor)


def test_xc_attention_forward_pass_without_cond(xc_attention_model):
    """ Test case to check if XCAttention handles forward pass without conditioning. """
    x = torch.randn(1, 256, 16, 16)

    output = xc_attention_model(x)

    assert isinstance(output, torch.Tensor)


def test_xc_attention_forward_with_invalid_inputs(xc_attention_model):
    """ Test case to check if XCAttention raises an error when forwarding with invalid inputs. """
    with pytest.raises(Exception):
        x = torch.randn(1, 256, 16, 16)
        cond = torch.randn(1, 128)  # Mismatched conditioning dimension
        xc_attention_model(x, cond)


def test_xc_attention_with_different_heads():
    """ Test case to check if XCAttention handles different head configurations correctly. """
    head_configs = [4, 8, 12]

    for heads in head_configs:
        model = XCAttention(dim=256, cond_dim=64, heads=heads)
        assert isinstance(model, XCAttention)
        assert (
            model.to_qkv[0].out_features
            == 3 * heads * model.norm.normalized_shape[0]
        )


def test_xc_attention_with_different_input_dims():
    """ Test case to check if XCAttention handles different input dimensions correctly. """
    input_dims = [128, 256, 512]

    for dim in input_dims:
        model = XCAttention(dim=dim, cond_dim=64, heads=8)
        assert isinstance(model, XCAttention)
        assert model.to_qkv[0].in_features == dim


def test_xc_attention_with_different_cond_dims():
    """ Test case to check if XCAttention handles different conditioning dimensions correctly. """
    cond_dims = [32, 64, 128]

    for cond_dim in cond_dims:
        model = XCAttention(dim=256, cond_dim=cond_dim, heads=8)
        assert isinstance(model, XCAttention)
        assert model.film[0].in_features == cond_dim * 2


def test_xc_attention_negative_input_dim():
    """ Test case to check if XCAttention handles negative input dimensions correctly. """
    with pytest.raises(ValueError):
        XCAttention(dim=-256, cond_dim=64, heads=8)


def test_xc_attention_negative_cond_dim():
    """ Test case to check if XCAttention handles negative conditioning dimensions correctly. """
    with pytest.raises(ValueError):
        XCAttention(dim=256, cond_dim=-64, heads=8)
