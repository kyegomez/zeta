import torch
import pytest
from zeta.nn.modules.visual_expert import (
    VisualExpert,)  # Import the VisualExpert class from your module


# Fixture for creating a sample instance of VisualExpert
@pytest.fixture
def visual_expert_instance():
    return VisualExpert(1024, 2048, 0.1, 16)


# Basic functionality tests
def test_visual_expert_creation(visual_expert_instance):
    assert isinstance(visual_expert_instance, VisualExpert)


def test_visual_expert_forward_pass(visual_expert_instance):
    x = torch.randn(1, 10, 1024)
    output = visual_expert_instance(x)
    assert output.shape == (1, 10, 1024)


# Parameterized tests for different input shapes and dimensions
@pytest.mark.parametrize("input_shape", [(1, 5, 1024), (2, 3, 1024)])
def test_visual_expert_parameterized(input_shape, visual_expert_instance):
    x = torch.randn(*input_shape)
    output = visual_expert_instance(x)
    assert output.shape == input_shape


# Test dropout rate
def test_visual_expert_dropout_rate(visual_expert_instance):
    assert visual_expert_instance.dropout == 0.1


# Test the number of attention heads
def test_visual_expert_attention_heads(visual_expert_instance):
    assert visual_expert_instance.heads == 16


# Test LayerNorm and Projections
def test_visual_expert_layers(visual_expert_instance):
    assert isinstance(visual_expert_instance.norm, torch.nn.LayerNorm)
    assert isinstance(visual_expert_instance.q_proj, torch.nn.Linear)
    assert isinstance(visual_expert_instance.k_proj, torch.nn.Linear)
    assert isinstance(visual_expert_instance.v_proj, torch.nn.Linear)


# Test attention and feedforward
def test_visual_expert_attention_and_feedforward(visual_expert_instance):
    assert isinstance(visual_expert_instance.attention,
                      torch.nn.modules.MultiheadAttention)
    assert isinstance(visual_expert_instance.feedforward,
                      torch.nn.modules.Linear)


# Test the call method with zero-sized input
def test_visual_expert_zero_input(visual_expert_instance):
    x = torch.empty(0, 10, 1024)
    output = visual_expert_instance(x)
    assert output.shape == (0, 10, 1024)


# Test the call method with negative values in the input tensor
def test_visual_expert_negative_input(visual_expert_instance):
    x = torch.randn(1, 10, 1024)
    x[x < 0] = -1
    output = visual_expert_instance(x)
    assert torch.all(output >= 0)


# Test that the forward pass maintains the shape
def test_visual_expert_shape_maintenance(visual_expert_instance):
    x = torch.randn(1, 10, 1024)
    initial_shape = x.shape
    output = visual_expert_instance(x)
    assert output.shape == initial_shape
