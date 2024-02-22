# tests/test_unet.py
import pytest
import torch

from zeta.nn.modules.unet import (  # Adjust this import according to your project structure
    Unet,
)


# Preparation of fixtures
@pytest.fixture
def n_channels():
    return 1


@pytest.fixture
def n_classes():
    return 2


@pytest.fixture
def input_tensor():
    return torch.randn(1, 1, 572, 572)


# Writing Basic Tests
def test_unet_initialization(n_channels, n_classes):
    model = Unet(n_channels, n_classes)
    assert model.n_channels == n_channels
    assert model.n_classes == n_classes
    assert not model.bilinear


def test_unet_forward_pass(n_channels, n_classes, input_tensor):
    model = Unet(n_channels, n_classes)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)


def test_unet_bilinear_option(n_channels, n_classes, input_tensor):
    model = Unet(n_channels, n_classes, bilinear=True)
    assert model.bilinear


# Utilize Fixtures
@pytest.fixture
def unet_model(n_channels, n_classes):
    return Unet(n_channels, n_classes)


def test_unet_output_shape(n_channels, n_classes, input_tensor, unet_model):
    output = unet_model(input_tensor)
    assert output.shape == (1, n_classes, 388, 388)


# Exception Testing
def test_unet_invalid_input_type():
    with pytest.raises(TypeError):
        Unet("invalid", "invalid")


# Parameterized Testing
@pytest.mark.parametrize(
    "n_channels, n_classes, expected_shape",
    [
        (1, 2, (1, 2, 388, 388)),
        (3, 4, (1, 4, 388, 388)),
        (5, 6, (1, 6, 388, 388)),
    ],
)
def test_unet_output_shape_with_parametrization(
    n_channels, n_classes, expected_shape, input_tensor
):
    model = Unet(n_channels, n_classes)
    output = model(input_tensor)
    assert output.shape == expected_shape


# Further tests would be added based on the full context and implementation details.
