import pytest
import torch

from zeta.structs import Transformer
from zeta.structs.transformer import AttentionLayers

# assuming that you are testing the Transformer class


# Start by initializing objects
@pytest.fixture()
def init_transformer():
    attn_layers = AttentionLayers(
        256
    )  # considering that AttentionLayers exist and received one parameter
    return Transformer(
        num_tokens=1000, max_seq_len=512, attn_layers=attn_layers
    )


# Basic tests: Like creating objects
def test_creation(init_transformer):
    transformer = init_transformer
    assert isinstance(transformer, Transformer)


# Parameterized Testing: Test if forward method is working as expected


@pytest.mark.parametrize(
    "x, expected_output_size",
    [
        (torch.randn(1, 512), (1, 1000)),
        (torch.randn(5, 256), (5, 1000)),
        (torch.randn(10, 200), (10, 1000)),
    ],
)
def test_forward(init_transformer, x, expected_output_size):
    output = init_transformer.forward(x)
    assert output.size() == expected_output_size


# Exception Testing: Check if errors are raised correctly
@pytest.mark.parametrize(
    "wrong_input", [torch.randn(1), torch.randn(1, 512, 3), "string"]
)
def test_forward_exception(init_transformer, wrong_input):
    with pytest.raises(ValueError):
        init_transformer.forward(wrong_input)
