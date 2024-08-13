import torch

from zeta.nn.quant.quick import QUIK


def test_quik_initialization():
    quik = QUIK(10, 20)

    assert isinstance(quik, QUIK)
    assert quik.in_features == 10
    assert quik.out_features == 20
    assert quik.quantize_range == 8
    assert quik.half_range == 4
    assert quik.weight.shape == (20, 10)
    assert quik.bias.shape == (20,)


def test_quik_quantize():
    quik = QUIK(10, 20)
    x = torch.randn(10, 10)
    quant_x, zero_act, scale_act = quik.quantize(x)

    assert isinstance(quant_x, torch.Tensor)
    assert quant_x.dtype == torch.int32
    assert isinstance(zero_act, torch.Tensor)
    assert isinstance(scale_act, torch.Tensor)


def test_quik_dequantize():
    quik = QUIK(10, 20)
    x = torch.randn(10, 10)
    quant_x, zero_act, scale_act = quik.quantize(x)
    dequant_x = quik.dequantize(quant_x, zero_act, scale_act, scale_act)

    assert isinstance(dequant_x, torch.Tensor)
    assert dequant_x.dtype == torch.float32


def test_quik_find_zero_scale():
    quik = QUIK(10, 20)
    x = torch.randn(10, 10)
    zero_act, scale_act = quik.find_zero_scale(x)

    assert isinstance(zero_act, torch.Tensor)
    assert isinstance(scale_act, torch.Tensor)


def test_quik_forward():
    quik = QUIK(10, 20)
    x = torch.randn(10, 10)
    output = quik(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (10, 20)
