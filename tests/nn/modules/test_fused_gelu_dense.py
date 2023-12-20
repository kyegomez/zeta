import pytest
import torch
from zeta.nn.modules.fused_gelu_dense import FusedDenseGELUDense


def test_class_init():
    model = FusedDenseGELUDense(512, 1024)

    assert model.dim == 512
    assert model.dim_out == 1024
    assert model.bias == True
    assert model.has_fp16_weights == False
    assert model.threshold == 6.0


def test_class_init_with_args():
    model = FusedDenseGELUDense(
        512, 1024, bias=False, has_fp16_weights=True, threshold=5.0
    )

    assert model.dim == 512
    assert model.dim_out == 1024
    assert model.bias == False
    assert model.has_fp16_weights == True
    assert model.threshold == 5.0


def test_forward():
    model = FusedDenseGELUDense(512, 1024)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_different_input():
    model = FusedDenseGELUDense(512, 1024)
    x = torch.randn(2, 512)
    out = model(x)

    assert out.shape == torch.Size([2, 512])


def test_forward_with_different_dim():
    model = FusedDenseGELUDense(256, 512)
    x = torch.randn(1, 256)
    out = model(x)

    assert out.shape == torch.Size([1, 256])


def test_forward_with_different_dim_out():
    model = FusedDenseGELUDense(512, 2048)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_no_bias():
    model = FusedDenseGELUDense(512, 1024, bias=False)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_fp16_weights():
    model = FusedDenseGELUDense(512, 1024, has_fp16_weights=True)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])


def test_forward_with_different_threshold():
    model = FusedDenseGELUDense(512, 1024, threshold=5.0)
    x = torch.randn(1, 512)
    out = model(x)

    assert out.shape == torch.Size([1, 512])
