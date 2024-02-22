import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from zeta.nn.attention.cross_attn_images import MultiModalCrossAttention


@pytest.fixture
def cross_attention_module():
    return MultiModalCrossAttention(1024, 8, 1024)


def test_forward_pass(cross_attention_module):
    input_dim = 1024
    seq_len = 32
    context_dim = 1024
    input_tensor = torch.randn(1, seq_len, input_dim)
    context_tensor = torch.randn(1, seq_len, context_dim)

    output = cross_attention_module(input_tensor, context_tensor)

    assert output.shape == (1, seq_len, input_dim)


def test_forward_pass_with_conditional_layer_norm(cross_attention_module):
    input_dim = 1024
    seq_len = 32
    context_dim = 1024
    input_tensor = torch.randn(1, seq_len, input_dim)
    context_tensor = torch.randn(1, seq_len, context_dim)

    cross_attention_module.qk = True  # Enable conditional layer normalization
    output = cross_attention_module(input_tensor, context_tensor)

    assert output.shape == (1, seq_len, input_dim)


def test_forward_pass_with_mask(cross_attention_module):
    input_dim = 1024
    seq_len = 32
    context_dim = 1024
    input_tensor = torch.randn(1, seq_len, input_dim)
    context_tensor = torch.randn(1, seq_len, context_dim)
    mask = torch.randint(0, 2, (seq_len, seq_len), dtype=torch.bool)

    cross_attention_module.mask = mask
    output = cross_attention_module(input_tensor, context_tensor)

    assert output.shape == (1, seq_len, input_dim)


def test_forward_pass_with_dropout(cross_attention_module):
    input_dim = 1024
    seq_len = 32
    context_dim = 1024
    input_tensor = torch.randn(1, seq_len, input_dim)
    context_tensor = torch.randn(1, seq_len, context_dim)

    cross_attention_module.dropout = nn.Dropout(0.5)  # Set dropout rate to 50%
    output = cross_attention_module(input_tensor, context_tensor)

    assert output.shape == (1, seq_len, input_dim)


def test_gradcheck(cross_attention_module):
    input_dim = 1024
    seq_len = 32
    context_dim = 1024
    input_tensor = torch.randn(1, seq_len, input_dim, requires_grad=True)
    context_tensor = torch.randn(1, seq_len, context_dim, requires_grad=True)

    assert gradcheck(
        cross_attention_module,
        (input_tensor, context_tensor),
        check_forward=True,
    )


def test_attention_strategy_average(cross_attention_module):
    input_dim = 1024
    seq_len = 32
    context_dim = 1024
    input_tensor = torch.randn(1, seq_len, input_dim)
    context_tensor = torch.randn(1, seq_len, context_dim)

    cross_attention_module.attention_strategy = "average"
    output = cross_attention_module(input_tensor, context_tensor)

    assert output.shape == (1, input_dim)


if __name__ == "__main__":
    pytest.main()
