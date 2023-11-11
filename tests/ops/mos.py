import torch
import pytest
from torch import nn
from zeta.ops.mos import (
    MixtureOfSoftmaxes,
)  # Replace 'your_module' with your actual module


# Create a fixture for initializing the model
@pytest.fixture
def mos_model():
    return MixtureOfSoftmaxes(num_mixtures=3, input_size=128, num_classes=10)


# Test basic functionality
def test_forward_pass(mos_model):
    input_data = torch.randn(32, 128)
    output = mos_model(input_data)
    assert output.shape == (32, 10)


# Test if model parameters are learnable
def test_parameter_update(mos_model):
    optimizer = torch.optim.SGD(mos_model.parameters(), lr=0.01)
    input_data = torch.randn(32, 128)
    target = torch.randint(10, (32,), dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(10):  # Training iterations
        optimizer.zero_grad()
        output = mos_model(input_data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    # Check if the model parameters have been updated
    for param in mos_model.parameters():
        assert param.grad is not None


# Test if the model handles different batch sizes
def test_different_batch_sizes(mos_model):
    batch_sizes = [16, 32, 64, 128]
    input_size = 128
    num_classes = 10

    for batch_size in batch_sizes:
        input_data = torch.randn(batch_size, input_size)
        output = mos_model(input_data)
        assert output.shape == (batch_size, num_classes)


# Test edge case with very large input size and number of classes
def test_large_input_and_classes():
    num_mixtures = 5
    input_size = 1024
    num_classes = 1000
    mos_model = MixtureOfSoftmaxes(num_mixtures, input_size, num_classes)
    input_data = torch.randn(64, input_size)
    output = mos_model(input_data)
    assert output.shape == (64, num_classes)


# Test if mixture weights sum to 1
def test_mixture_weights_sum_to_one(mos_model):
    input_data = torch.randn(32, 128)
    mixture_weights = mos_model.mixture_weights(input_data)
    assert torch.allclose(mixture_weights.sum(dim=1), torch.ones(32), atol=1e-5)


# Test if softmax outputs sum to 1
def test_softmax_outputs_sum_to_one(mos_model):
    input_data = torch.randn(32, 128)
    output = mos_model(input_data)
    assert torch.allclose(output.sum(dim=1), torch.ones(32), atol=1e-5)


# Test if mixture weights are within [0, 1]
def test_mixture_weights_range(mos_model):
    input_data = torch.randn(32, 128)
    mixture_weights = mos_model.mixture_weights(input_data)
    assert torch.all(mixture_weights >= 0) and torch.all(mixture_weights <= 1)


# Test if softmax outputs are within [0, 1]
def test_softmax_outputs_range(mos_model):
    input_data = torch.randn(32, 128)
    output = mos_model(input_data)
    assert torch.all(output >= 0) and torch.all(output <= 1)


# Test edge case with zero input size and classes
def test_zero_input_size_and_classes():
    mos_model = MixtureOfSoftmaxes(num_mixtures=2, input_size=0, num_classes=0)
    input_data = torch.randn(32, 0)
    output = mos_model(input_data)
    assert output.shape == (32, 0)


# Test if mixture weights are uniform when input is zero
def test_uniform_mixture_weights_on_zero_input(mos_model):
    input_data = torch.zeros(32, 128)
    mixture_weights = mos_model.mixture_weights(input_data)
    assert torch.allclose(mixture_weights, torch.ones(32, 3) / 3, atol=1e-5)


# Test if mixture weights are non-uniform when input is constant
def test_non_uniform_mixture_weights_on_constant_input(mos_model):
    input_data = torch.ones(32, 128)
    mixture_weights = mos_model.mixture_weights(input_data)
    assert not torch.allclose(mixture_weights, torch.ones(32, 3) / 3, atol=1e-5)


# Test if the model handles large number of mixtures
def test_large_num_mixtures():
    num_mixtures = 100
    input_size = 128
    num_classes = 10
    mos_model = MixtureOfSoftmaxes(num_mixtures, input_size, num_classes)
    input_data = torch.randn(32, input_size)
    output = mos_model(input_data)
    assert output.shape == (32, num_classes)


# Test if the model handles very small number of mixtures
def test_small_num_mixtures():
    num_mixtures = 1
    input_size = 128
    num_classes = 10
    mos_model = MixtureOfSoftmaxes(num_mixtures, input_size, num_classes)
    input_data = torch.randn(32, input_size)
    output = mos_model(input_data)
    assert output.shape == (32, num_classes)


# Test if the model handles very small input data
def test_small_input_data():
    num_mixtures = 3
    input_size = 1
    num_classes = 10
    mos_model = MixtureOfSoftmaxes(num_mixtures, input_size, num_classes)
    input_data = torch.randn(32, input_size)
    output = mos_model(input_data)
    assert output.shape == (32, num_classes)


# Test if the model handles large input data
def test_large_input_data():
    num_mixtures = 3
    input_size = 2048
    num_classes = 10
    mos_model = MixtureOfSoftmaxes(num_mixtures, input_size, num_classes)
    input_data = torch.randn(32, input_size)
    output = mos_model(input_data)
    assert output.shape == (32, num_classes)
