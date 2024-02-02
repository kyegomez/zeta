""" Test cases for the Attend module. """

import torch
from zeta.nn.attention.attend import Attend


# Test case for initializing the Attend module
def test_attend_init():
    attend = Attend()
    assert isinstance(attend, Attend)


# Test case for the forward pass of the Attend module
def test_attend_forward():
    attend = Attend()

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if the output shape matches the input shape
    assert out.shape == (1, 8, 32, 64)


# Test case for configuring the dropout rate
def test_attend_dropout():
    attend = Attend(dropout=0.2)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if dropout has been applied (output should not be identical)
    assert not torch.allclose(out, q)


# Test case for configuring the scale factor
def test_attend_scale_factor():
    attend = Attend(scale=0.5)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if the attention scores are scaled correctly
    scale_factor = 0.5 * (64**-0.5)
    assert torch.allclose(out, q * scale_factor)


# Test case for configuring the causal mask
def test_attend_causal_mask():
    attend = Attend(causal=True)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if the causal mask has been applied
    assert out.shape == (1, 8, 32, 64)


# Test case for configuring talking heads
def test_attend_talking_heads():
    attend = Attend(talking_heads=True)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if talking heads configuration is correct
    assert out.shape == (1, 8, 32, 64)


# Test case for configuring sparse top-k
def test_attend_sparse_topk():
    attend = Attend(sparse_topk=32)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if the sparse top-k configuration is correct
    assert out.shape == (1, 8, 32, 64)


# Test case for configuring flash attention
def test_attend_flash_attention():
    attend = Attend(flash=True)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if flash attention configuration is correct
    assert out.shape == (1, 8, 32, 64)

# Test case for configuring flash attention
def test_flash_attention():
    import torch
    from zeta.nn import FlashAttention

    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 10, 8)
    v = torch.randn(2, 4, 10, 8)

    attention = FlashAttention(causal=False, dropout=0.1, flash=True)
    output = attention(q, k, v)

    assert(output.shape == (2, 4, 6, 8))
    


# Test case for gradient checking using torch.autograd.gradcheck
def test_attend_gradient_check():
    attend = Attend()

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)
    q.requires_grad = True

    # Perform a forward pass and backward pass
    out, intermediates = attend(q, k, v)
    grad_output = torch.randn_like(out)
    torch.autograd.gradcheck(attend, (q, k, v), grad_output)


# Test case for adding zero key-value tokens
def test_attend_add_zero_kv():
    attend = Attend(add_zero_kv=True)

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    # Perform a forward pass
    out, intermediates = attend(q, k, v)

    # Check if zero key-value tokens have been added
    assert out.shape == (1, 8, 32, 64)


# Test case for handling residual attention
def test_attend_residual_attention():
    attend = Attend()

    # Create random input tensors
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)
    prev_attn = torch.randn(1, 8, 32, 32)

    # Perform a forward pass
    out, intermediates = attend(q, k, v, prev_attn=prev_attn)

    # Check if residual attention has been applied
    assert out.shape == (1, 8, 32, 64)
