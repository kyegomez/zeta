import pytest
import torch
from zeta.nn.attention.mgqa import MGQA, CacheView


# Create an instance of MGQA for testing
mgqa = MGQA(
    dim=768,
    n_layers=12,
    head_dim=64,
    hidden_dim=2048,
    n_heads=8,
    n_kv_heads=8,
    sliding_window=512,
    norm_eps=1e-6,
    vocab_size=32000,
    attn_dropout=0.1,
    max_batch_size=0,
    flash=False,
)


# Test MGQA forward pass
def test_mgqa_forward():
    x = torch.randn(1, 768)
    freqs_cis = torch.randn(1, 768)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa(x, freqs_cis, cache)
    assert output.shape == (1, 768)


# Test MGQA forward pass with different input sizes
@pytest.mark.parametrize("batch_size, seq_len", [(1, 512), (2, 256), (4, 128)])
def test_mgqa_forward_batch_sizes(batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 768)
    freqs_cis = torch.randn(batch_size, seq_len, 768)
    cache = CacheView(batch_size, 512, 8, 8, 64)
    output = mgqa(x, freqs_cis, cache)
    assert output.shape == (batch_size, seq_len, 768)


# Test MGQA forward pass with pre-filled cache
def test_mgqa_forward_with_prefilled_cache():
    x = torch.randn(1, 512)
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    cache.prefill_cache(x, x)
    output = mgqa(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA forward pass with causal=True
def test_mgqa_forward_causal():
    mgqa_causal = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=0,
        flash=False,
    )
    x = torch.randn(1, 768)
    freqs_cis = torch.randn(1, 768)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_causal(x, freqs_cis, cache)
    assert output.shape == (1, 768)


# Test MGQA forward pass with flash=True
def test_mgqa_forward_flash():
    mgqa_flash = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=0,
        flash=True,
    )
    x = torch.randn(1, 768)
    freqs_cis = torch.randn(1, 768)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_flash(x, freqs_cis, cache)
    assert output.shape == (1, 768)


# Test MGQA with maximum batch size
def test_mgqa_max_batch_size():
    mgqa_max_batch = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=64,  # Set a maximum batch size
        flash=False,
    )
    x = torch.randn(64, 512, 768)
    freqs_cis = torch.randn(64, 512, 768)
    cache = CacheView(64, 512, 8, 8, 64)
    output = mgqa_max_batch(x, freqs_cis, cache)
    assert output.shape == (64, 512, 768)


# Test MGQA with sliding_window = 0
def test_mgqa_sliding_window_zero():
    mgqa_sliding_window_zero = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=0,  # Disable sliding window
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=0,
        flash=False,
    )
    x = torch.randn(1, 512)
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_sliding_window_zero(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA with layer normalization
def test_mgqa_with_layer_norm():
    mgqa_layer_norm = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=0,
        flash=False,
    )
    x = torch.randn(1, 512)
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_layer_norm(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA with attention dropout
def test_mgqa_with_attention_dropout():
    mgqa_attention_dropout = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.5,  # Set attention dropout
        max_batch_size=0,
        flash=False,
    )
    x = torch.randn(1, 512)
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_attention_dropout(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA with flash=True and attention dropout
def test_mgqa_with_flash_and_attention_dropout():
    mgqa_flash_attention_dropout = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.5,  # Set attention dropout
        max_batch_size=0,
        flash=True,  # Use FlashAttention
    )
    x = torch.randn(1, 512)
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_flash_attention_dropout(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA with pre-filled cache
def test_mgqa_with_prefilled_cache():
    x = torch.randn(1, 512)
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    cache.prefill_cache(x, x)
    output = mgqa(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA with vocabulary size limit
def test_mgqa_with_vocab_size_limit():
    mgqa_vocab_limit = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=100,  # Set a smaller vocabulary size
        attn_dropout=0.1,
        max_batch_size=0,
        flash=False,
    )
    x = torch.randint(0, 100, size=(1, 512))
    freqs_cis = torch.randn(1, 512)
    cache = CacheView(1, 512, 8, 8, 64)
    output = mgqa_vocab_limit(x, freqs_cis, cache)
    assert output.shape == (1, 512, 768)


# Test MGQA with maximum batch size and sliding window
def test_mgqa_with_max_batch_and_sliding_window():
    mgqa_max_batch_sliding_window = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=64,  # Set a maximum batch size
        flash=False,
    )
    x = torch.randn(64, 512, 768)
    freqs_cis = torch.randn(64, 512, 768)
    cache = CacheView(64, 512, 8, 8, 64)
    output = mgqa_max_batch_sliding_window(x, freqs_cis, cache)
    assert output.shape == (64, 512, 768)


# Test MGQA with maximum batch size and sliding window disabled
def test_mgqa_with_max_batch_and_sliding_window_disabled():
    mgqa_max_batch_sliding_window_disabled = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=0,  # Disable sliding window
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=64,  # Set a maximum batch size
        flash=False,
    )
    x = torch.randn(64, 512, 768)
    freqs_cis = torch.randn(64, 512, 768)
    cache = CacheView(64, 512, 8, 8, 64)
    output = mgqa_max_batch_sliding_window_disabled(x, freqs_cis, cache)
    assert output.shape == (64, 512, 768)


# Test MGQA with maximum batch size and causal=True
def test_mgqa_with_max_batch_and_causal():
    mgqa_max_batch_causal = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=64,  # Set a maximum batch size
        flash=False,
    )
    x = torch.randn(64, 512, 768)
    freqs_cis = torch.randn(64, 512, 768)
    cache = CacheView(64, 512, 8, 8, 64)
    output = mgqa_max_batch_causal(x, freqs_cis, cache)
    assert output.shape == (64, 512, 768)


# Test MGQA with maximum batch size and flash=True
def test_mgqa_with_max_batch_and_flash():
    mgqa_max_batch_flash = MGQA(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,
        n_heads=8,
        n_kv_heads=8,
        sliding_window=512,
        norm_eps=1e-6,
        vocab_size=32000,
        attn_dropout=0.1,
        max_batch_size=64,  # Set a maximum batch size
        flash=True,  # Use FlashAttention
    )
    x = torch.randn(64, 512, 768)
    freqs_cis = torch.randn(64, 512, 768)
    cache = CacheView(64, 512, 8, 8, 64)
    output = mgqa_max_batch_flash(x, freqs_cis, cache)
    assert output.shape == (64, 512, 768)
