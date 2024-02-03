# GELUActivation

import math
import pytest
import torch
from zeta.nn import GELUActivation


# Basic functionality tests
@pytest.mark.parametrize(
    "input, expected_output",
    [
        (torch.tensor([0.0]), torch.tensor([0.0])),
        (
            torch.tensor([1.0]),
            torch.tensor([0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))]),
        ),
    ],
)
def test_gelu_activation_forward_method(input, expected_output):
    gelu = GELUActivation(use_gelu_python=True)
    assert torch.allclose(gelu.forward(input), expected_output, atol=1e-6)


# Test for checking if PyTorch's GELU is used when use_gelu_python is False
def test_gelu_activation_with_pytorch_gelu():
    gelu = GELUActivation(use_gelu_python=False)
    input = torch.tensor([1.0])
    assert torch.allclose(
        gelu.forward(input), torch.nn.functional.gelu(input), atol=1e-6
    )


# Edge cases
def test_gelu_activation_with_large_positive_input():
    gelu = GELUActivation(use_gelu_python=True)
    input = torch.tensor([10000.0])
    assert torch.allclose(gelu.forward(input), input, atol=1e-6)


def test_gelu_activation_with_large_negative_input():
    gelu = GELUActivation(use_gelu_python=True)
    input = torch.tensor([-10000.0])
    assert torch.allclose(gelu.forward(input), torch.tensor([-0.0]), atol=1e-6)


# Error handling
def test_gelu_activation_with_invalid_input():
    gelu = GELUActivation(use_gelu_python=True)
    with pytest.raises(TypeError):
        _ = gelu.forward("not a tensor")
