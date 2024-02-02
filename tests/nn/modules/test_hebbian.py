import pytest
import torch

from zeta.nn.modules.hebbian import (
    BasicHebbianGRUModel,)  # Import your module here


# Fixture for creating an instance of the model
@pytest.fixture
def model_instance():
    input_dim = 512
    hidden_dim = 256
    output_dim = 128
    model = BasicHebbianGRUModel(input_dim, hidden_dim, output_dim)
    return model


# Test case for model instantiation
def test_model_instantiation(model_instance):
    assert isinstance(model_instance, BasicHebbianGRUModel)


# Test case for forward pass with random input
def test_forward_pass(model_instance):
    batch_size = 32
    seqlen = 10
    input_dim = 512
    input_tensor = torch.randn(batch_size, seqlen, input_dim)
    output = model_instance(input_tensor)
    assert output.shape == (batch_size, seqlen, model_instance.output_dim)


# Test case for weights initialization
def test_weights_initialization(model_instance):
    for param in model_instance.parameters():
        if param.requires_grad:
            assert torch.all(param != 0.0)


# Test case for input dimension matching
def test_input_dimension_matching(model_instance):
    input_tensor = torch.randn(16, 20, 512)
    with pytest.raises(RuntimeError):
        _ = model_instance(input_tensor)


# Test case for output dimension matching
def test_output_dimension_matching(model_instance):
    input_tensor = torch.randn(16, 20, 512)
    output = model_instance(input_tensor)
    assert output.shape == (16, 20, model_instance.output_dim)


# Add more test cases to thoroughly cover your module's functionality
