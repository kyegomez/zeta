# DenseBlock

import torch
import torch.nn as nn
import pytest

from zeta.nn import DenseBlock


def test_DenseBlock_init():
    conv = nn.Conv2d(1, 20, 5)
    dense_block = DenseBlock(conv)
    assert dense_block.submodule == conv, "Submodule not initialized correctly."


def test_DenseBlock_forward():
    conv = nn.Conv2d(1, 20, 5)
    dense_block = DenseBlock(conv)
    x = torch.randn(1, 1, 24, 24)
    output = dense_block(x)
    assert output.shape == torch.Size(
        [1, 21, 20, 20]
    ), "Forward function not working properly."


@pytest.mark.parametrize("invalid_submodule", [None, 5, "invalid", []])
def test_DenseBlock_init_invalid_submodule(invalid_submodule):
    with pytest.raises(TypeError):
        DenseBlock(invalid_submodule)


@pytest.mark.parametrize("invalid_input", [None, 5, "invalid", []])
def test_DenseBlock_forward_invalid_input(invalid_input):
    conv = nn.Conv2d(1, 20, 5)
    dense_block = DenseBlock(conv)
    with pytest.raises(Exception):
        dense_block(invalid_input)
