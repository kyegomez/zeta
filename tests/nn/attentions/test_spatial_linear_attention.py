import torch
import torch.nn as nn
from zeta.nn.attention.spatial_linear_attention import SpatialLinearAttention


def test_spatial_linear_attention_init():
    sla = SpatialLinearAttention(dim=64, heads=4, dim_head=16)
    assert isinstance(sla, SpatialLinearAttention)
    assert sla.scale == 16**-0.5
    assert sla.heads == 4
    assert isinstance(sla.to_qkv, nn.Conv2d)
    assert isinstance(sla.to_out, nn.Conv2d)


def test_spatial_linear_attention_forward():
    sla = SpatialLinearAttention(dim=64, heads=4, dim_head=16)
    x = torch.randn(2, 64, 10, 32, 32)
    output = sla.forward(x)
    assert output.shape == (2, 64, 10, 32, 32)


def test_spatial_linear_attention_forward_zero_input():
    sla = SpatialLinearAttention(dim=64, heads=4, dim_head=16)
    x = torch.zeros(2, 64, 10, 32, 32)
    output = sla.forward(x)
    assert output.shape == (2, 64, 10, 32, 32)
    assert torch.all(output == 0)


def test_spatial_linear_attention_forward_one_input():
    sla = SpatialLinearAttention(dim=64, heads=4, dim_head=16)
    x = torch.ones(2, 64, 10, 32, 32)
    output = sla.forward(x)
    assert output.shape == (2, 64, 10, 32, 32)
