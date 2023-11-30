import pytest
import torch
from torch import nn
from zeta.nn.modules.expert import (
    Experts,
)  # Import the Experts class from your module


# Define fixtures
@pytest.fixture
def experts_model():
    return Experts(512, 16)


# Test parameter initialization and correctness of shapes
def test_experts_parameter_initialization(experts_model):
    assert isinstance(experts_model.w1, nn.Parameter)
    assert isinstance(experts_model.w2, nn.Parameter)
    assert isinstance(experts_model.w3, nn.Parameter)
    assert experts_model.w1.shape == (16, 512, 1024)
    assert experts_model.w2.shape == (16, 2048, 2048)
    assert experts_model.w3.shape == (16, 2048, 512)


# Test forward pass
def test_experts_forward_pass(experts_model):
    batch_size, seq_len, dim = 1, 3, 512
    x = torch.randn(batch_size, seq_len, dim)
    out = experts_model(x)
    assert out.shape == (batch_size, seq_len, dim)


# Test activation function
def test_experts_activation_function(experts_model):
    batch_size, seq_len, dim = 1, 3, 512
    x = torch.randn(batch_size, seq_len, dim)
    out = experts_model(x)
    assert torch.all(out >= 0)  # Ensure non-negative values


# Test input validation
def test_experts_input_validation():
    with pytest.raises(ValueError):
        Experts(512, -16)  # Negative number of experts should raise an error


# Test documentation examples
def test_documentation_examples():
    x = torch.randn(1, 3, 512)
    model = Experts(512, 16)
    out = model(x)
    assert out.shape == (1, 3, 512)


# Parameterized testing for various input sizes
@pytest.mark.parametrize(
    "batch_size, seq_len, dim, experts",
    [
        (1, 3, 512, 16),
        (2, 4, 256, 8),
        (3, 5, 128, 4),
    ],
)
def test_experts_parameterized(batch_size, seq_len, dim, experts):
    x = torch.randn(batch_size, seq_len, dim)
    model = Experts(dim, experts)
    out = model(x)
    assert out.shape == (batch_size, seq_len, dim)


# Test if the LeakyReLU activation function is used
def test_experts_activation_function_used(experts_model):
    assert any(
        isinstance(module, nn.LeakyReLU) for module in experts_model.modules()
    )


# Test if the expert weights are learnable parameters
def test_experts_weights_learnable(experts_model):
    assert any(param.requires_grad for param in experts_model.parameters())


# More extensive testing can be added as needed, following the same pattern
# ...


# Test edge cases
def test_experts_edge_cases():
    # Test with minimal input size
    model = Experts(1, 1)
    x = torch.randn(1, 1, 1)
    out = model(x)
    assert out.shape == (1, 1, 1)

    # Test with zero-dimensional input
    model = Experts(0, 1)
    x = torch.empty(0, 0, 0)
    out = model(x)
    assert out.shape == (0, 0, 0)
