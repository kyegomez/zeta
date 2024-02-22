import torch
import torch.nn as nn

from zeta.quant.half_bit_linear import HalfBitLinear


def test_half_bit_linear_init():
    hbl = HalfBitLinear(10, 5)
    assert isinstance(hbl, HalfBitLinear)
    assert hbl.in_features == 10
    assert hbl.out_features == 5
    assert isinstance(hbl.weight, nn.Parameter)
    assert isinstance(hbl.bias, nn.Parameter)


def test_half_bit_linear_forward():
    hbl = HalfBitLinear(10, 5)
    x = torch.randn(1, 10)
    output = hbl.forward(x)
    assert output.shape == (1, 5)


def test_half_bit_linear_forward_zero_input():
    hbl = HalfBitLinear(10, 5)
    x = torch.zeros(1, 10)
    output = hbl.forward(x)
    assert output.shape == (1, 5)
    assert torch.all(output == 0)


def test_half_bit_linear_forward_one_input():
    hbl = HalfBitLinear(10, 5)
    x = torch.ones(1, 10)
    output = hbl.forward(x)
    assert output.shape == (1, 5)
