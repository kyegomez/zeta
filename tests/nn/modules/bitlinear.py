import pytest
import torch
from torch import nn
from zeta.quant.bitlinear import absmax_quantize, BitLinear

def test_absmax_quantize():
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    quant, dequant = absmax_quantize(x)

    assert isinstance(quant, torch.Tensor)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, x, atol=1e-2)

@pytest.mark.parametrize("bits", [4, 8, 16])
def test_absmax_quantize_different_bits(bits):
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    quant, dequant = absmax_quantize(x, bits)

    assert isinstance(quant, torch.Tensor)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, x, atol=1e-2)

def test_bitlinear_init():
    bitlinear = BitLinear(10, 20)

    assert isinstance(bitlinear, nn.Module)
    assert bitlinear.in_features == 10
    assert bitlinear.out_features == 20
    assert bitlinear.groups == 1
    assert isinstance(bitlinear.weight, nn.Parameter)

def test_bitlinear_forward():
    bitlinear = BitLinear(10, 20)
    input = torch.randn(128, 10)
    output = bitlinear(input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (128, 20)

@pytest.mark.parametrize("groups", [1, 2, 4])
def test_bitlinear_different_groups(groups):
    bitlinear = BitLinear(10, 20, groups)
    input = torch.randn(128, 10)
    output = bitlinear(input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (128, 20)