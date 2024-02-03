# FeedbackBlock

# Import necessary libraries
import pytest
import torch
import torch.nn as nn
from zeta.nn import FeedbackBlock


# Set up simple neural network module for testing FeedbackBlock
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


# Define fixture for FeedbackBlock instance with TestModule
@pytest.fixture
def feedback_block():
    return FeedbackBlock(TestModule())


def test_initialization(feedback_block):
    assert isinstance(feedback_block, FeedbackBlock)
    assert isinstance(feedback_block.submodule, TestModule)


@pytest.mark.parametrize(
    "input_tensor,feedback_tensor,expected_output_shape",
    [
        (
            torch.rand(1, 10),
            torch.rand(1, 10),
            (1, 10),
        ),  # Test with valid input and feedback tensors
        (
            torch.rand(1, 10),
            None,
            (1, 10),
        ),  # Test with valid input and no feedback
        (
            torch.rand(1, 10),
            torch.rand(1, 20),
            pytest.raises(ValueError),
        ),  # Test with mismatching dimension
    ],
)
def test_forward(
    feedback_block, input_tensor, feedback_tensor, expected_output_shape
):
    if isinstance(expected_output_shape, tuple):
        assert (
            feedback_block.forward(input_tensor, feedback_tensor).shape
            == expected_output_shape
        )
    else:
        with expected_output_shape:
            feedback_block.forward(input_tensor, feedback_tensor)
