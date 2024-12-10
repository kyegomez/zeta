# NewGELUActivation

import math

import pytest
import torch
from torch import Tensor, nn

from zeta.nn import NewGELUActivation


def test_newgeluactivation_instance():
    gelu = NewGELUActivation()
    assert isinstance(gelu, nn.Module)


def test_newgeluactivation_forward_valid_tensor():
    gelu = NewGELUActivation()
    test_tensor = torch.randn(3, 3)
    out = gelu.forward(test_tensor)
    assert out.size() == test_tensor.size()


def test_newgeluactivation_forward_return_type():
    gelu = NewGELUActivation()
    test_tensor = torch.randn(3, 3)
    out = gelu.forward(test_tensor)
    assert isinstance(out, Tensor)


def test_newgeluactivation_forward_value_range():
    gelu = NewGELUActivation()
    test_tensor = torch.randn(3, 3)
    out = gelu.forward(test_tensor)
    assert out.min() >= 0
    assert out.max() <= 1


@pytest.mark.parametrize("test_input,expected", [(-1, 0), (0, 0), (1, 1)])
def test_newgeluactivation_forward_values(test_input, expected):
    gelu = NewGELUActivation()
    test_tensor = torch.tensor([test_input], dtype=torch.float32)
    out = gelu.forward(test_tensor)
    assert math.isclose(out.item(), expected, rel_tol=1e-7)


def test_newgeluactivation_forward_handle_empty():
    gelu = NewGELUActivation()
    with pytest.raises(RuntimeError):
        gelu.forward(torch.tensor([]))


def test_newgeluactivation_forward_handle_none():
    gelu = NewGELUActivation()
    with pytest.raises(TypeError):
        gelu.forward(None)


def test_newgeluactivation_forward_handle_string():
    gelu = NewGELUActivation()
    with pytest.raises(TypeError):
        gelu.forward("string")
