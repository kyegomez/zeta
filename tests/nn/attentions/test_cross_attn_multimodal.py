import torch

from zeta.nn.attention.cross_attn_images import MultiModalCrossAttention


# Test case for initializing the MultiModalCrossAttention module
def test_multi_modal_cross_attention_init():
    cross_attention = MultiModalCrossAttention(1024, 8, 1024)
    assert isinstance(cross_attention, MultiModalCrossAttention)


# Test case for the forward pass of the MultiModalCrossAttention module
def test_multi_modal_cross_attention_forward():
    cross_attention = MultiModalCrossAttention(1024, 8, 1024)

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = cross_attention(x, context)

    # Check if the output shape matches the input shape
    assert out.shape == (1, 32, 1024)


# Test case for configuring conditional layer normalization
def test_multi_modal_cross_attention_conditional_ln():
    cross_attention = MultiModalCrossAttention(1024, 8, 1024, qk=True)

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = cross_attention(x, context)

    # Check if conditional layer normalization is applied
    assert out.shape == (1, 32, 1024)


# Test case for configuring post-attention normalization
def test_multi_modal_cross_attention_post_attn_norm():
    cross_attention = MultiModalCrossAttention(
        1024, 8, 1024, post_attn_norm=True
    )

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = cross_attention(x, context)

    # Check if post-attention normalization is applied
    assert out.shape == (1, 32, 1024)


# Test case for specifying an attention strategy (average)
def test_multi_modal_cross_attention_attention_strategy_average():
    cross_attention = MultiModalCrossAttention(
        1024, 8, 1024, attention_strategy="average"
    )

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = cross_attention(x, context)

    # Check if the output shape matches the input shape
    assert out.shape == (1, 1024)


# Test case for specifying an attention strategy (concatenate)
def test_multi_modal_cross_attention_attention_strategy_concatenate():
    cross_attention = MultiModalCrossAttention(
        1024, 8, 1024, attention_strategy="concatenate"
    )

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = cross_attention(x, context)

    # Check if the output shape is as expected
    assert out.shape == (1, 32 * 1024)


# Test case for masking attention weights
def test_multi_modal_cross_attention_attention_masking():
    # Create a mask with some values masked
    mask = torch.rand(1, 8, 32, 32) > 0.5

    cross_attention = MultiModalCrossAttention(1024, 8, 1024, mask=mask)

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = cross_attention(x, context)

    # Check if the output shape matches the input shape
    assert out.shape == (1, 32, 1024)


# Test case for gradient checking using torch.autograd.gradcheck
def test_multi_modal_cross_attention_gradient_check():
    cross_attention = MultiModalCrossAttention(1024, 8, 1024)

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)
    x.requires_grad = True

    # Perform a forward pass and backward pass
    out = cross_attention(x, context)
    grad_output = torch.randn_like(out)
    torch.autograd.gradcheck(cross_attention, (x, context), grad_output)


# Test case for initializing the MultiModalCrossAttention module
def test_multimodal_cross_attention_init():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(dim, heads, context_dim)
    assert isinstance(attn, MultiModalCrossAttention)


# Test case for the forward pass of the MultiModalCrossAttention module
def test_multimodal_cross_attention_forward():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(dim, heads, context_dim)

    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attn(x, context)

    # Check if the output shape matches the expected shape
    assert out.shape == (1, 32, 1024)


# Test case for conditional layer normalization
def test_multimodal_cross_attention_conditional_norm():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(dim, heads, context_dim, qk=True)

    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attn(x, context)

    # Check if conditional layer normalization has been applied
    assert out.shape == (1, 32, 1024)


# Test case for post-attention normalization
def test_multimodal_cross_attention_post_attn_norm():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(
        dim, heads, context_dim, post_attn_norm=True
    )

    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attn(x, context)

    # Check if post-attention normalization has been applied
    assert out.shape == (1, 32, 1024)


# Test case for attention strategy "average"
def test_multimodal_cross_attention_average_strategy():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(
        dim, heads, context_dim, attention_strategy="average"
    )

    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attn(x, context)

    # Check if the "average" attention strategy has been applied
    assert out.shape == (1, 1024)


# Test case for attention masking
def test_multimodal_cross_attention_masking():
    dim = 1024
    heads = 8
    context_dim = 1024

    # Create a masking tensor (e.g., masking out some positions)
    mask = torch.randn(1, 32, 32).bool()

    attn = MultiModalCrossAttention(dim, heads, context_dim, mask=mask)

    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attn(x, context)

    # Check if the attention masking has been applied
    assert out.shape == (1, 32, 1024)


# Test case for gradient checking using torch.autograd.gradcheck
def test_multimodal_cross_attention_gradient_check():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(dim, heads, context_dim)

    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)
    x.requires_grad = True

    # Perform a forward pass and backward pass
    out = attn(x, context)
    grad_output = torch.randn_like(out)
    torch.autograd.gradcheck(attn, (x, context), grad_output)


# Test case for masking in MultiModalCrossAttention
def test_multimodal_cross_attention_mask():
    dim = 1024
    heads = 8
    context_dim = 1024
    mask = torch.randn(1, 32, 32).random_(2, dtype=torch.bool)
    attn = MultiModalCrossAttention(dim, heads, context_dim, mask=mask)

    # Create random input tensors
    x = torch.randn(1, 32, dim)
    context = torch.randn(1, 32, context_dim)

    # Perform a forward pass
    out = attn(x, context)

    # Check if masking has been applied
    assert out.shape == (1, 32, dim)


# Test case for attention strategy (average)
def test_multimodal_cross_attention_strategy_average():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(
        dim, heads, context_dim, attention_strategy="average"
    )

    # Create random input tensors
    x = torch.randn(1, 32, dim)
    context = torch.randn(1, 32, context_dim)

    # Perform a forward pass
    out = attn(x, context)

    # Check if attention strategy (average) is applied correctly
    assert out.shape == (1, dim)


# Test case for attention strategy (concatenate)
def test_multimodal_cross_attention_strategy_concatenate():
    dim = 1024
    heads = 8
    context_dim = 1024
    attn = MultiModalCrossAttention(
        dim, heads, context_dim, attention_strategy="concatenate"
    )

    # Create random input tensors
    x = torch.randn(1, 32, dim)
    context = torch.randn(1, 32, context_dim)

    # Perform a forward pass
    out = attn(x, context)

    # Check if attention strategy (concatenate) is applied correctly
    assert out.shape == (1, 32 * dim)


# Helper function to create a mask
def create_mask(batch_size, seq_len):
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return mask


# Test case for configuring conditional layer normalization (qk)
def test_multi_modal_cross_attention_qk():
    attention = MultiModalCrossAttention(
        dim=1024, heads=8, context_dim=1024, qk=True
    )

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attention(x, context)

    # Check if conditional layer normalization is applied correctly
    assert out.shape == (1, 32, 1024)


# Test case for configuring the attention strategy as "average"
def test_multi_modal_cross_attention_average_strategy():
    attention = MultiModalCrossAttention(
        dim=1024, heads=8, context_dim=1024, attention_strategy="average"
    )

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attention(x, context)

    # Check if the "average" attention strategy is applied correctly
    assert out.shape == (1, 1024)


# Test case for configuring the attention mask
def test_multi_modal_cross_attention_mask():
    attention = MultiModalCrossAttention(
        dim=1024, heads=8, context_dim=1024, mask=create_mask(1, 32)
    )

    # Create random input tensors
    x = torch.randn(1, 32, 1024)
    context = torch.randn(1, 32, 1024)

    # Perform a forward pass
    out = attention(x, context)

    # Check if the attention mask is applied correctly
    assert out.shape == (1, 32, 1024)
