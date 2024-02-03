# QuickGELUActivation

import pytest
import torch
from zeta.nn import QuickGELUActivation


@pytest.fixture
def quick_gelu_activation():
    return QuickGELUActivation()


def test_initialization(quick_gelu_activation):
    assert isinstance(quick_gelu_activation, QuickGELUActivation)


def test_forward_pass_zero(quick_gelu_activation):
    input_tensor = torch.tensor([0.0])
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert output_tensor.item() == 0.0


def test_forward_pass_positive(quick_gelu_activation):
    input_tensor = torch.tensor([1.0])
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert output_tensor.item() > 0.0


def test_forward_pass_negative(quick_gelu_activation):
    input_tensor = torch.tensor([-1.0])
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert output_tensor.item() < 0.0


@pytest.mark.parametrize(
    "input_tensor", [torch.tensor([2.0]), torch.tensor([-2.0])]
)
def test_forward_pass_greater_than_one(quick_gelu_activation, input_tensor):
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert abs(output_tensor.item()) > abs(input_tensor.item())


def test_forward_pass_non_tensor(quick_gelu_activation):
    input_data = [1, 2, 3]
    with pytest.raises(TypeError):
        quick_gelu_activation.forward(input_data)


def test_forward_pass_empty_tensor(quick_gelu_activation):
    input_tensor = torch.tensor([])
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert len(output_tensor) == 0.0


def test_forward_pass_1d_tensor(quick_gelu_activation):
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_forward_pass_2d_tensor(quick_gelu_activation):
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output_tensor = quick_gelu_activation.forward(input_tensor)
    assert output_tensor.shape == input_tensor.shape
