# ReLUSquaredActivation

import pytest
import torch
from zeta.nn import ReLUSquaredActivation


def test_relu_squared_activation_instance():
    layer = ReLUSquaredActivation()
    assert isinstance(layer, ReLUSquaredActivation)


def test_relu_squared_activation_forward():
    layer = ReLUSquaredActivation()
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    output_tensor = layer.forward(input_tensor)
    expected_output = torch.tensor([0.0, 0.0, 1.0, 4.0])  # Relu Squared Output
    assert torch.equal(output_tensor, expected_output)


@pytest.mark.parametrize(
    "input_tensor, expected_output",
    [
        (
            torch.tensor([-1.0, 0.0, 1.0, 2.0]),
            torch.tensor([0.0, 0.0, 1.0, 4.0]),
        ),
        (
            torch.tensor([3.0, -3.0, 3.0, -3.0]),
            torch.tensor([9.0, 0.0, 9.0, 0.0]),
        ),
    ],
)
def test_relu_squared_activation_parametrized(input_tensor, expected_output):
    layer = ReLUSquaredActivation()
    output_tensor = layer.forward(input_tensor)
    assert torch.equal(output_tensor, expected_output)


def test_relu_squared_activation_exception():
    layer = ReLUSquaredActivation()
    with pytest.raises(TypeError):
        layer.forward("Invalid input")


def test_relu_squared_activation_negative_values():
    layer = ReLUSquaredActivation()
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    output_tensor = layer.forward(input_tensor)
    assert (
        torch.sum(output_tensor) == 0
    )  # All negative values should be relu'd to zero, and then squared to zero
