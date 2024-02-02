# ClippedGELUActivation

import pytest
from unittest.mock import Mock, patch
import torch
from torch import Tensor
from zeta.nn import ClippedGELUActivation


# Assume gelu function is in same module for simplicity
def gelu(x: Tensor):
    return (0.5 * x * (1 + torch.tanh(
        torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3)))))


# Test if ValueError is raised when min > max
def test_initialization_error():
    with pytest.raises(ValueError) as err:
        ClippedGELUActivation(2.0, 1.0)
    assert str(err.value) == "min should be < max (got min: 2.0, max: 1.0)"


# Test forward function with mock GELU function
def test_forward():
    mock = Mock(spec=gelu)
    mock.return_value = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    with patch("zeta.nn.gelu", new=mock):
        act_func = ClippedGELUActivation(-0.5, 1.5)
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0])
        result = act_func.forward(x)
        mock.assert_called_once_with(x)
        assert torch.all(result.eq(torch.tensor([-0.5, 0.0, 1.0, 1.5])))


# Test parametrized inputs
@pytest.mark.parametrize(
    "input_tensor, output_tensor",
    [
        (
            torch.tensor([-1.0, 0.0, 1.0, 2.0]),
            torch.tensor([-0.5, 0.0, 0.5, 1.0]),
        ),
        (
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
        ),
        (
            torch.tensor([2.0, -2.0, -2.0, 2.0]),
            torch.tensor([1.0, -1.0, -1.0, 1.0]),
        ),
    ],
)
def test_forward_parametrized(input_tensor, output_tensor):
    act_func = ClippedGELUActivation(-1.0, 1.0)
    result = act_func.forward(input_tensor)
    assert torch.all(result.eq(output_tensor))
