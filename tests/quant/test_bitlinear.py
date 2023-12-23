import pytest
import torch
from torch import nn
from zeta.quant.bitlinear import BitLinear, absmax_quantize


def test_bitlinear_reset_parameters():
    bitlinear = BitLinear(10, 20)
    old_weight = bitlinear.weight.clone()
    bitlinear.reset_parameters()

    assert not torch.equal(old_weight, bitlinear.weight)


def test_bitlinear_forward_quantization():
    bitlinear = BitLinear(10, 20)
    input = torch.randn(128, 10)
    output = bitlinear(input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (128, 20)

    # Check that the output is different from the input, indicating that quantization and dequantization occurred
    assert not torch.allclose(output, input)


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_absmax_quantize_different_bits(bits):
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    quant, dequant = absmax_quantize(x, bits)

    assert isinstance(quant, torch.Tensor)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, x, atol=1e-2)

    # Check that the quantized values are within the expected range
    assert quant.min() >= -(2 ** (bits - 1))
    assert quant.max() <= 2 ** (bits - 1) - 1
