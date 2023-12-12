import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta.nn.modules.polymorphic_neuron import PolymorphicNeuronLayer


# Fixture for creating a sample PolymorphicNeuronLayer instance
@pytest.fixture
def sample_neuron():
    return PolymorphicNeuronLayer(in_features=10, out_features=5)


# Basic initialization test
def test_neuron_initialization(sample_neuron):
    assert isinstance(sample_neuron, PolymorphicNeuronLayer)
    assert sample_neuron.in_features == 10
    assert sample_neuron.out_features == 5
    assert isinstance(sample_neuron.weights, nn.Parameter)
    assert isinstance(sample_neuron.bias, nn.Parameter)


# Test forward pass with a sample input
def test_forward_pass(sample_neuron):
    input_tensor = torch.randn(1, 10)
    output = sample_neuron(input_tensor)
    assert output.shape == (1, 5)


# Parameterized test for different activation functions
@pytest.mark.parametrize("activation", [F.relu, F.tanh, F.sigmoid])
def test_different_activation_functions(activation):
    neuron = PolymorphicNeuronLayer(
        in_features=10, out_features=5, activation_functions=[activation]
    )
    input_tensor = torch.randn(1, 10)
    output = neuron(input_tensor)
    assert output.shape == (1, 5)


# Test for a case where input features and output features are both 0
def test_zero_features():
    with pytest.raises(ValueError):
        PolymorphicNeuronLayer(in_features=0, out_features=0)


# Test for a case where the activation functions list is empty
def test_empty_activation_functions():
    with pytest.raises(ValueError):
        PolymorphicNeuronLayer(
            in_features=10, out_features=5, activation_functions=[]
        )


# Test for a case where in_features and out_features are negative
def test_negative_features():
    with pytest.raises(ValueError):
        PolymorphicNeuronLayer(in_features=-10, out_features=-5)


# Test for a case where input tensor shape does not match in_features
def test_input_tensor_shape_mismatch(sample_neuron):
    input_tensor = torch.randn(1, 5)  # Mismatched input shape
    with pytest.raises(ValueError):
        sample_neuron(input_tensor)


# Test for a case where activation functions are not callable
def test_invalid_activation_functions():
    with pytest.raises(ValueError):
        PolymorphicNeuronLayer(
            in_features=10, out_features=5, activation_functions=[1, 2, 3]
        )


# Test for a case where the forward pass is called without initializing weights and bias
def test_forward_pass_without_initialization():
    neuron = PolymorphicNeuronLayer(in_features=10, out_features=5)
    input_tensor = torch.randn(1, 10)
    with pytest.raises(RuntimeError):
        neuron(input_tensor)


# Test if all the activation functions in the list are used at least once
def test_all_activation_functions_used(sample_neuron):
    input_tensor = torch.randn(1, 10)
    output = sample_neuron(input_tensor)
    unique_activations = set(output.unique().numpy())
    assert len(unique_activations) == len(sample_neuron.activation_functions)


# Test that forward pass results are within valid range
def test_output_range(sample_neuron):
    input_tensor = torch.randn(1, 10)
    output = sample_neuron(input_tensor)
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
