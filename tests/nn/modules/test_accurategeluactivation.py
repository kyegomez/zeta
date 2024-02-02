# AccurateGELUActivation

# 1. Importing necessary libraries
import math
import pytest
import torch
from zeta.nn import AccurateGELUActivation


# 2. Basic Test
def test_init():
    activation = AccurateGELUActivation()
    assert activation.precomputed_constant == math.sqrt(2 / math.pi)


# 3. Testing Forward Operation
def test_forward():
    activation = AccurateGELUActivation()
    input_data = torch.Tensor([1.0, 2.0, 3.0])
    result = activation.forward(input_data)
    assert torch.is_tensor(result)


# Parameterized Testing
@pytest.mark.parametrize("input_data", [([1.0, 2.0, 3.0]), ([-1.0, -2.0, -3.0]),
                                        ([0.0, 0.0, 0.0])])
def test_forward_parameterized(input_data):
    activation = AccurateGELUActivation()
    input_data = torch.Tensor(input_data)
    result = activation.forward(input_data)
    assert torch.is_tensor(result)


# Exception Testing
def test_forward_exception():
    activation = AccurateGELUActivation()
    with pytest.raises(TypeError):
        activation.forward("Invalid input")


# Mocks and Monkeypatching
def test_forward_monkeypatch(monkeypatch):

    def mock_tanh(x):
        return torch.Tensor([0.0 for _ in x])

    monkeypatch.setattr(torch, "tanh", mock_tanh)
    activation = AccurateGELUActivation()
    input_data = torch.Tensor([1.0, 2.0, 3.0])
    result = activation.forward(input_data)
    assert result.equal(torch.Tensor([0.0, 1.0, 1.5]))

    monkeypatch.undo()
