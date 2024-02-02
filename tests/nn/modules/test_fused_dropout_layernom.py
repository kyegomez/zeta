import torch
from torch import nn
from zeta.nn.modules.fused_dropout_layernom import FusedDropoutLayerNorm


def test_class_init():
    model = FusedDropoutLayerNorm(512)

    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.layer_norm, nn.LayerNorm)


def test_class_init_with_args():
    model = FusedDropoutLayerNorm(512,
                                  dropout=0.2,
                                  eps=1e-6,
                                  elementwise_affine=False)

    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.layer_norm, nn.LayerNorm)
    assert model.dropout.p == 0.2
    assert model.layer_norm.eps == 1e-6
    assert model.layer_norm.elementwise_affine is False


def test_forward():
    model = FusedDropoutLayerNorm(512)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_different_input():
    model = FusedDropoutLayerNorm(512)
    x = torch.randn(2, 512)
    out = model(x)

    assert out.shape == torch.Size([2, 512])


def test_forward_with_different_dim():
    model = FusedDropoutLayerNorm(256)
    x = torch.randn(1, 256)
    out = model(x)

    assert out.shape == torch.Size([1, 256])


def test_forward_with_different_dropout():
    model = FusedDropoutLayerNorm(512, dropout=0.2)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_different_eps():
    model = FusedDropoutLayerNorm(512, eps=1e-6)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_no_elementwise_affine():
    model = FusedDropoutLayerNorm(512, elementwise_affine=False)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])
