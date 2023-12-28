import torch
import torch.nn as nn
import pytest
from zeta.nn.modules.dense_connect import DenseBlock


@pytest.fixture
def dense_block():
    submodule = nn.Linear(10, 5)
    return DenseBlock(submodule)


def test_forward(dense_block):
    x = torch.randn(32, 10)
    output = dense_block(x)

    assert output.shape == (32, 15)  # Check output shape
    assert torch.allclose(output[:, :10], x)  # Check if input is preserved
    assert torch.allclose(
        output[:, 10:], dense_block.submodule(x)
    )  # Check submodule output


def test_initialization(dense_block):
    assert isinstance(dense_block.submodule, nn.Linear)  # Check submodule type
    assert dense_block.submodule.in_features == 10  # Check input features
    assert dense_block.submodule.out_features == 5  # Check output features


def test_docstrings():
    assert (
        DenseBlock.__init__.__doc__ is not None
    )  # Check if __init__ has a docstring
    assert (
        DenseBlock.forward.__doc__ is not None
    )  # Check if forward has a docstring
