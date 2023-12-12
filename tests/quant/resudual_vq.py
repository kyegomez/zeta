import torch
import torch.nn as nn
from zeta.quant.residual_vq import ResidualVectorQuantizer


def test_residual_vector_quantizer_init():
    model = ResidualVectorQuantizer(4, 4, 4)
    assert isinstance(model, nn.Module)
    assert model.dim == 4
    assert model.dim_out == 4
    assert model.n_embed == 4
    assert isinstance(model.embed, nn.Embedding)
    assert isinstance(model.proj, nn.Linear)


def test_residual_vector_quantizer_forward():
    model = ResidualVectorQuantizer(4, 4, 4)
    x = torch.randn(2, 4)
    out = model(x)
    assert out.shape == torch.Size([2, 4])


def test_residual_vector_quantizer_forward_zero():
    model = ResidualVectorQuantizer(4, 4, 4)
    x = torch.zeros(2, 4)
    out = model(x)
    assert torch.all(out == 0)


def test_residual_vector_quantizer_forward_one():
    model = ResidualVectorQuantizer(4, 4, 4)
    x = torch.ones(2, 4)
    out = model(x)
    assert torch.all(out == 1)
