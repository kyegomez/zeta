import torch
from zeta.quant.absmax import absmax_quantize


def test_absmax_quantize_default_bits():
    x = torch.randn(128)
    quant, dequant = absmax_quantize(x)
    assert quant.dtype == torch.int8
    assert dequant.dtype == torch.float32
    assert torch.allclose(dequant, x, atol=1e-2)


def test_absmax_quantize_custom_bits():
    x = torch.randn(128)
    quant, dequant = absmax_quantize(x, bits=16)
    assert quant.dtype == torch.int8
    assert dequant.dtype == torch.float32
    assert torch.allclose(dequant, x, atol=1e-4)


def test_absmax_quantize_zero_tensor():
    x = torch.zeros(128)
    quant, dequant = absmax_quantize(x)
    assert torch.all(quant == 0)
    assert torch.all(dequant == 0)


def test_absmax_quantize_positive_tensor():
    x = torch.ones(128)
    quant, dequant = absmax_quantize(x)
    assert torch.all(quant == 2**7 - 1)
    assert torch.allclose(dequant, x, atol=1e-4)


def test_absmax_quantize_negative_tensor():
    x = -torch.ones(128)
    quant, dequant = absmax_quantize(x)
    assert torch.all(quant == -(2**7 - 1))
    assert torch.allclose(dequant, x, atol=1e-4)
