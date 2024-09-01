import torch
import torch.nn as nn

from zeta.nn.modules.quantized_layernorm import QuantizedLN


def test_quantized_ln_init():
    ln = QuantizedLN(10)
    assert isinstance(ln, QuantizedLN)
    assert ln.bits == 8
    assert isinstance(ln.ln, nn.LayerNorm)


def test_quantized_ln_forward():
    ln = QuantizedLN(10)
    x = torch.randn(128, 10)
    output = ln(x)
    assert output.shape == x.shape


def test_quantized_ln_bits():
    ln = QuantizedLN(10, bits=16)
    assert ln.bits == 16


def test_quantized_ln_eps():
    ln = QuantizedLN(10, eps=1e-3)
    assert ln.ln.eps == 1e-3


def test_quantized_ln_elementwise_affine():
    ln = QuantizedLN(10, element_wise_affine=False)
    assert ln.ln.elementwise_affine is False


def test_quantized_ln_normalized_shape():
    ln = QuantizedLN((128, 10))
    x = torch.randn(128, 10)
    output = ln(x)
    assert output.shape == x.shape
