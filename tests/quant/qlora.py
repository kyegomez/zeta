import pytest
import torch
import torch.nn as nn
from torch.testing import assert_allclose
from zeta.quant.qlora import QloraLinear

# Sample instantiation values
in_features = 20
out_features = 30
weight = torch.randn(out_features, in_features)
r = 5
lora_alpha = 2
lora_dropout = 0.5


@pytest.fixture
def qlora_layer():
    return QloraLinear(in_features, out_features, weight, r, lora_alpha, lora_dropout)


def test_initialization(qlora_layer):
    assert qlora_layer.in_features == in_features
    assert qlora_layer.out_features == out_features
    assert qlora_layer.r == r
    assert qlora_layer.lora_alpha == lora_alpha
    assert qlora_layer.scaling == lora_alpha / r


def test_reset_parameters(qlora_layer):
    qlora_layer.reset_parameters()
    assert not torch.all(qlora_layer.lora_B == 0)


@pytest.mark.parametrize(
    "input_tensor", [torch.randn(128, in_features), torch.randn(1, in_features)]
)
def test_forward_pass_shape(qlora_layer, input_tensor):
    output = qlora_layer(input_tensor)
    assert output.shape == (input_tensor.shape[0], out_features)


def test_forward_pass_calculation(qlora_layer):
    input_tensor = torch.randn(128, in_features)
    output = qlora_layer(input_tensor)
    base_output = input_tensor @ weight.transpose(0, 1)
    lora_output = (
        input_tensor @ qlora_layer.lora_A.transpose(0, 1)
    ) @ qlora_layer.lora_B.transpose(0, 1)
    expected_output = base_output + lora_output * qlora_layer.scaling
    assert_allclose(output, expected_output, atol=1e-4)


def test_lora_dropout(qlora_layer):
    qlora_layer.lora_dropout.p = 1.0  # set dropout to 100%
    input_tensor = torch.randn(128, in_features)
    output = qlora_layer(input_tensor)
    base_output = input_tensor @ weight.transpose(0, 1)
    assert_allclose(output, base_output, atol=1e-4)


def test_invalid_input_shape(qlora_layer):
    with pytest.raises(ValueError):
        qlora_layer(torch.randn(128, in_features + 1))
