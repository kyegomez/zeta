# HighwayLayer

import pytest
import torch
import torch.nn as nn
from zeta.nn import HighwayLayer


def test_highway_layer_init():
    """
    Tests for HighwayLayer's __init__ function.
    """
    layer = HighwayLayer(10)

    assert isinstance(layer, nn.Module)
    assert isinstance(layer.normal_layer, nn.Linear)
    assert isinstance(layer.gate, nn.Linear)
    assert layer.normal_layer.in_features == 10

    # test for exception handling
    with pytest.raises(TypeError):
        layer = HighwayLayer("invalid_dim")


@pytest.mark.parametrize(
    "dim, input_value, expected_dim",
    [(5, [1, 2, 3, 4, 5], (5,)), (3, [[1, 2, 3], [4, 5, 6]], (2, 3))],
)
def test_highway_layer_forward(dim, input_value, expected_dim):
    """
    Test for HighwayLayer's forward function.
    """
    layer = HighwayLayer(dim)
    tensor_input = torch.tensor(input_value, dtype=torch.float32)
    tensor_output = layer.forward(tensor_input)

    # Check output type and dim
    assert isinstance(tensor_output, torch.Tensor)
    assert tensor_output.shape == expected_dim
    assert tensor_output.dtype == torch.float32


@pytest.mark.parametrize("dim", [(5), (10), (15)])
def test_highway_layer_with_different_dim(dim):
    """
    Test for HighwayLayer with different dim in the __init__ function.
    """
    layer = HighwayLayer(dim)
    assert layer.normal_layer.in_features == dim
    assert layer.gate.in_features == dim


@pytest.mark.parametrize("data_type", [(torch.float16), (torch.float64)])
def test_highway_layer_with_different_data_types(data_type):
    """
    Test for HighwayLayer with different data types of input tensor in the forward function
    """
    layer = HighwayLayer(5)
    tensor_input = torch.tensor([1, 2, 3, 4, 5], dtype=data_type)
    tensor_output = layer.forward(tensor_input)
    assert tensor_output.dtype == data_type
