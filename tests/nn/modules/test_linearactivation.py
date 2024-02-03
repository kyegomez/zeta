# LinearActivation

import torch
import pytest
from zeta.nn import LinearActivation


def test_LinearActivation_init():
    assert isinstance(LinearActivation(), LinearActivation)


@pytest.mark.parametrize(
    "input_tensor", [(torch.tensor([1, 2, 3])), (torch.tensor([-1, 0, 1]))]
)
def test_LinearActivation_forward(input_tensor):
    """Test if the forward method of LinearActivation class returns the same input tensor."""
    act = LinearActivation()
    assert torch.equal(act.forward(input_tensor), input_tensor)


def test_LinearActivation_forward_error():
    """Test if the forward method of LinearActivation class raises an error when input tensor is not valid."""
    act = LinearActivation()
    with pytest.raises(TypeError):
        invalid_input = [1, 2, "a"]
        act.forward(torch.tensor(invalid_input))
