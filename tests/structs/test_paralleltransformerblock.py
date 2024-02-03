import torch
import pytest
from zeta.structs import ParallelTransformerBlock
from torch.autograd import gradcheck


# Basic Testing
def test_parallel_transformer_block_init():
    p = ParallelTransformerBlock(512)
    assert p.fused_dims == (512, 64, 64, 2048)
    assert p.scale == 1 / (64**0.5)


def test_parallel_transformer_block_forward():
    p = ParallelTransformerBlock(512)
    x = torch.randn(1, 10, 512)
    output = p(x)
    assert output.size() == (1, 10, 512)


# Parameterized Testing
@pytest.mark.parametrize(
    "dim, dim_head, heads, ff_mult", [(128, 16, 4, 6), (256, 32, 8, 3)]
)
def test_parallel_transformer_block_param(dim, dim_head, heads, ff_mult):
    p = ParallelTransformerBlock(dim, dim_head, heads, ff_mult)
    assert isinstance(p, ParallelTransformerBlock)


# Exception Testing
def test_invalid_input():
    p = ParallelTransformerBlock(512)
    x = torch.randn(1, 512)  # Should be a 3D tensor
    with pytest.raises(Exception):
        p(x)


# Fixture usage
@pytest.fixture
def parallel_transformer_block():
    return ParallelTransformerBlock(512)


def test_forward_with_fixture(parallel_transformer_block):
    input = torch.randn(1, 10, 512, requires_grad=True)
    output = parallel_transformer_block(input)
    assert output.size() == (1, 10, 512)


# Tests for Mask and Position Embedding
def test_mask_functionality(parallel_transformer_block):
    mask_output = parallel_transformer_block.get_mask(10, torch.device("cpu"))
    assert mask_output.shape == (10, 10)


def test_rotary_embedding_functionality(parallel_transformer_block):
    pos_emb_output = parallel_transformer_block.get_rotary_embedding(
        10, torch.device("cpu")
    )
    assert pos_emb_output.shape == (10, 8)


# Gradients and Parameter testing
def test_gradient(parallel_transformer_block):
    input = torch.randn(1, 10, 512, requires_grad=True)
    # Check the gradients pass
    assert gradcheck(parallel_transformer_block, input, eps=1e-6, atol=1e-4)
