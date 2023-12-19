import pytest
import torch
import torch.nn as nn
from zeta.nn.modules.flexible_mlp import CustomMLP


# Fixture for creating a sample CustomMLP instance
@pytest.fixture
def sample_mlp():
    return CustomMLP(layer_sizes=[10, 5, 2], activation="relu", dropout=0.5)


# Basic initialization test
def test_mlp_initialization(sample_mlp):
    assert isinstance(sample_mlp, CustomMLP)
    assert isinstance(sample_mlp.layers, nn.ModuleList)
    assert callable(sample_mlp.activation_fn)
    assert sample_mlp.dropout.p == 0.5


# Test forward pass with a sample input
def test_forward_pass(sample_mlp):
    input_tensor = torch.randn(1, 10)
    output = sample_mlp(input_tensor)
    assert output.shape == (1, 2)


# Parameterized testing for different layer sizes
@pytest.mark.parametrize(
    "layer_sizes",
    [
        [10, 5, 2],
        [5, 3, 1],
        [20, 10, 5],
    ],
)
def test_different_layer_sizes(layer_sizes):
    mlp = CustomMLP(layer_sizes=layer_sizes)
    input_tensor = torch.randn(1, layer_sizes[0])
    output = mlp(input_tensor)
    assert output.shape == (1, layer_sizes[-1])


# Test for an unsupported activation function
def test_unsupported_activation():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10, 5, 2], activation="invalid_activation")


# Test for negative dropout probability
def test_negative_dropout():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10, 5, 2], dropout=-0.1)


# Test for dropout probability greater than 1.0
def test_large_dropout():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10, 5, 2], dropout=1.1)


# Test for empty layer_sizes list
def test_empty_layer_sizes():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[])


# Test for a single-layer MLP
def test_single_layer_mlp():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10])


# Test dropout functionality
def test_dropout(sample_mlp):
    # Check if dropout is applied by checking the output shape
    input_tensor = torch.randn(1, 10)
    output = sample_mlp(input_tensor)
    assert output.shape == (1, 2)


# Parameterized test for different activation functions
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_different_activation_functions(activation):
    mlp = CustomMLP(layer_sizes=[10, 5, 2], activation=activation, dropout=0.0)
    input_tensor = torch.randn(1, 10)
    output = mlp(input_tensor)
    assert output.shape == (1, 2)


# Test for invalid layer_sizes input
def test_invalid_layer_sizes():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[], activation="relu", dropout=0.0)


# Test for invalid layer_sizes input (less than 2 elements)
def test_invalid_layer_sizes_length():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10], activation="relu", dropout=0.0)


# Test for invalid layer_sizes input (negative elements)
def test_invalid_layer_sizes_negative():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10, -5, 2], activation="relu", dropout=0.0)


# Test for invalid dropout input (greater than 1)
def test_invalid_dropout():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10, 5, 2], activation="relu", dropout=1.5)


# Test for invalid dropout input (less than 0)
def test_invalid_dropout_negative():
    with pytest.raises(ValueError):
        CustomMLP(layer_sizes=[10, 5, 2], activation="relu", dropout=-0.5)


# Test for unsupported activation function
def test_invalid_activation_function():
    with pytest.raises(ValueError):
        CustomMLP(
            layer_sizes=[10, 5, 2], activation="invalid_activation", dropout=0.0
        )


# Additional tests related to edge cases and boundary conditions can be added as needed
