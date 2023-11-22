import torch
import torch.nn as nn
import pytest
from zeta.nn.modules.alr_block import FeedForward, ALRBlock


# Create fixtures
@pytest.fixture
def sample_input():
    return torch.randn(1, 1024, 512)


@pytest.fixture
def alrblock_model():
    return ALRBlock(512, 2048, 0.1)


@pytest.fixture
def feedforward_model():
    return FeedForward(512, 2048, 0.1)


# Tests for FeedForward class
def test_feedforward_creation():
    model = FeedForward(512, 2048, 0.1)
    assert isinstance(model, nn.Module)


def test_feedforward_forward(sample_input, feedforward_model):
    output = feedforward_model(sample_input)
    assert output.shape == sample_input.shape


# Tests for ALRBlock class
def test_alrblock_creation(alrblock_model):
    assert isinstance(alrblock_model, nn.Module)


def test_alrblock_forward(sample_input, alrblock_model):
    output = alrblock_model(sample_input)
    assert output.shape == sample_input.shape


# Parameterized testing for various input dimensions and dropout rates
@pytest.mark.parametrize(
    "input_dim, hidden_dim, dropout",
    [
        (256, 1024, 0.2),
        (512, 2048, 0.0),
        (128, 512, 0.3),
    ],
)
def test_feedforward_parameterized(input_dim, hidden_dim, dropout):
    model = FeedForward(input_dim, hidden_dim, dropout)
    input_tensor = torch.randn(1, 1024, input_dim)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


@pytest.mark.parametrize(
    "dim, hidden_dim, dropout",
    [
        (256, 1024, 0.2),
        (512, 2048, 0.0),
        (128, 512, 0.3),
    ],
)
def test_alrblock_parameterized(dim, hidden_dim, dropout):
    model = ALRBlock(dim, hidden_dim, dropout)
    input_tensor = torch.randn(1, 1024, dim)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Exception testing
def test_feedforward_invalid_input():
    model = FeedForward(512, 2048, 0.1)
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 1024, 512))  # Invalid batch size


def test_alrblock_invalid_input():
    model = ALRBlock(512, 2048, 0.1)
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 1024, 512))  # Invalid batch size
