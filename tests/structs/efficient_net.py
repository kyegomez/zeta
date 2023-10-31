import pytest
import torch
import torch.nn as nn
from zeta.structs import EfficientNet


@pytest.fixture
def model():
    return EfficientNet()


def test_model_creation(model):
    assert isinstance(model, EfficientNet)


def test_forward_pass(model):
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_5D_input(model):
    x = torch.randn(1, 5, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 5, 1000)


def test_forward_pass_with_different_input_shape(model):
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    assert output.shape == (2, 1000)


def test_forward_pass_with_different_width_mult(model):
    model = EfficientNet(width_mult=0.5)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_5D_input_and_different_width_mult(model):
    model = EfficientNet(width_mult=0.5)
    x = torch.randn(1, 5, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 5, 1000)


def test_forward_pass_with_different_input_shape_and_width_mult(model):
    model = EfficientNet(width_mult=0.5)
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    assert output.shape == (2, 1000)


def test_forward_pass_with_large_input_shape(model):
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_5D_input_and_large_input_shape(model):
    x = torch.randn(1, 5, 3, 512, 512)
    output = model(x)
    assert output.shape == (1, 5, 1000)


def test_forward_pass_with_different_input_shape_and_large_input_shape(model):
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    assert output.shape == (2, 1000)


def test_forward_pass_with_zero_input(model):
    x = torch.zeros(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_negative_input(model):
    x = torch.randn(1, 3, 256, 256) * -1
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_inf_input(model):
    x = torch.randn(1, 3, 256, 256)
    x[0, 0, 0, 0] = float("inf")
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_nan_input(model):
    x = torch.randn(1, 3, 256, 256)
    x[0, 0, 0, 0] = float("nan")
    output = model(x)
    assert output.shape == (1, 1000)


def test_forward_pass_with_large_output_shape(model):
    x = torch.randn(1, 3, 256, 256)
    model.classifier = nn.Linear(1280, 10000)
    output = model(x)
    assert output.shape == (1, 10000)


def test_forward_pass_with_5D_input_and_large_output_shape(model):
    x = torch.randn(1, 5, 3, 256, 256)
    model.classifier = nn.Linear(1280, 10000)
    output = model(x)
    assert output.shape == (1, 5, 10000)


def test_forward_pass_with_different_input_shape_and_large_output_shape(model):
    x = torch.randn(2, 3, 256, 256)
    model.classifier = nn.Linear(1280, 10000)
    output = model(x)
    assert output.shape == (2, 10000)
