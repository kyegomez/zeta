import pytest
import torch
from zeta.nn.modules.simple_feedforward import (
    SimpleFeedForward,
)  # Adjust import as per your project structure


# Fixture for creating a SimpleFeedForward model
@pytest.fixture
def model():
    return SimpleFeedForward(dim=768, hidden_dim=2048, dropout=0.1)


# Fixture for creating a sample input tensor
@pytest.fixture
def input_tensor():
    return torch.randn(1, 768)


# Test to check if model returns tensor with expected shape
def test_model_shape(model, input_tensor):
    output = model(input_tensor)
    assert output.shape == torch.Size([1, 768])


# Test to check if dropout is applied (output should be different with each forward pass)
def test_dropout_effect(model, input_tensor):
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    assert not torch.equal(output1, output2)


# Test to check if model can handle different input dimensions
@pytest.mark.parametrize("dim", [256, 512, 1024])
def test_different_input_dimensions(dim):
    model = SimpleFeedForward(dim=dim, hidden_dim=2048, dropout=0.1)
    input_tensor = torch.randn(1, dim)
    output = model(input_tensor)
    assert output.shape == torch.Size([1, dim])


# Test to check if model handles zero dropout correctly (output should be same with each forward pass)
def test_zero_dropout(model, input_tensor):
    model_no_dropout = SimpleFeedForward(dim=768, hidden_dim=2048, dropout=0)
    output1 = model_no_dropout(input_tensor)
    output2 = model_no_dropout(input_tensor)
    assert torch.equal(output1, output2)


# Test to check if model handles invalid input dimensions
def test_invalid_input_dimensions():
    with pytest.raises(ValueError):
        model = SimpleFeedForward(dim=-1, hidden_dim=2048, dropout=0.1)


# ... (continue adding more test cases as per the guide)
