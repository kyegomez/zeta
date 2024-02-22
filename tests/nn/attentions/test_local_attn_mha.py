import pytest
import torch
from torch.autograd import gradcheck

from zeta.nn.attention.local_attention_mha import LocalMHA

# Create an instance of LocalMHA for testing
local_mha = LocalMHA(
    dim=256,
    window_size=32,
    dim_head=64,
    heads=8,
    dropout=0.1,
    causal=False,
    prenorm=False,
    qk_rmsnorm=False,
    qk_scale=8,
    use_xpos=False,
    xpos_scale_base=None,
    exact_windowsize=None,
)


# Helper function to generate random input data
def generate_random_input(batch_size, seq_len, emb_dim):
    return torch.randn(batch_size, seq_len, emb_dim)


# Helper function to check if a tensor is sparse (contains zeros)
def is_sparse(tensor):
    return (tensor == 0).all()


# Test the forward pass of LocalMHA
def test_local_mha_forward():
    batch_size = 4
    seq_len = 32
    emb_dim = 256

    input_data = generate_random_input(batch_size, seq_len, emb_dim)
    output = local_mha(input_data)
    assert output.shape == (batch_size, seq_len, emb_dim)


# Test LocalMHA with different heads
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_local_mha_with_different_heads(heads):
    local_mha = LocalMHA(
        dim=256,
        window_size=32,
        dim_head=64,
        heads=heads,
        dropout=0.1,
        causal=False,
        prenorm=False,
        qk_rmsnorm=False,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
    )

    batch_size = 4
    seq_len = 32
    emb_dim = 256

    input_data = generate_random_input(batch_size, seq_len, emb_dim)
    output = local_mha(input_data)
    assert output.shape == (batch_size, seq_len, emb_dim)


# Test LocalMHA with different window sizes
@pytest.mark.parametrize("window_size", [16, 32, 64, 128])
def test_local_mha_with_different_window_sizes(window_size):
    local_mha = LocalMHA(
        dim=256,
        window_size=window_size,
        dim_head=64,
        heads=8,
        dropout=0.1,
        causal=False,
        prenorm=False,
        qk_rmsnorm=False,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
    )

    batch_size = 4
    seq_len = 32
    emb_dim = 256

    input_data = generate_random_input(batch_size, seq_len, emb_dim)
    output = local_mha(input_data)
    assert output.shape == (batch_size, seq_len, emb_dim)


# Test if the output of LocalMHA is sparse
def test_local_mha_output_sparse():
    batch_size = 4
    seq_len = 32
    emb_dim = 256

    input_data = torch.zeros(
        batch_size, seq_len, emb_dim
    )  # Create a tensor with all zeros
    output = local_mha(input_data)
    assert is_sparse(output)  # Check if the output is sparse


# Test gradient checking for LocalMHA
def test_local_mha_gradient_check():
    batch_size = 4
    seq_len = 32
    emb_dim = 256

    input_data = generate_random_input(batch_size, seq_len, emb_dim)
    input_data.requires_grad = True

    gradcheck(local_mha, (input_data,), raise_exception=True)
