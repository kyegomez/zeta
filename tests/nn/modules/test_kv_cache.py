from unittest.mock import Mock

import pytest
import torch

from zeta.nn.modules.kv_cache import (
    find_multiple,
    precompute_freq_cis,
    KVCache,
    setup_cache,
)


# 1. Basic Tests
def test_find_multiple():
    assert find_multiple(10, 3) == 12
    assert find_multiple(15, 5) == 15
    assert find_multiple(20, 7) == 21


def test_precompute_freq_cis():
    seq_len = 128
    n_elem = 64
    freqs = precompute_freq_cis(seq_len, n_elem)
    assert freqs.shape == torch.Size([seq_len, n_elem, 2])


def test_kv_cache_creation():
    cache = KVCache(32, 128, 8, 64)
    assert isinstance(cache, KVCache)


# 2. Utilize Fixtures
@pytest.fixture
def sample_cache():
    return KVCache(16, 64, 4, 32)


def test_kv_cache_update(sample_cache):
    input_pos = torch.randint(0, 64, (5,))
    k_val = torch.randn(16, 4, 64, 32)
    v_val = torch.randn(16, 4, 64, 32)
    k_out, v_out = sample_cache.update(input_pos, k_val, v_val)
    assert k_out.shape == torch.Size([16, 4, 64, 32])
    assert v_out.shape == torch.Size([16, 4, 64, 32])


# 3. Parameterized Testing
@pytest.mark.parametrize(
    "max_batch_size, max_seq_len, heads, head_dim",
    [(32, 128, 8, 64), (16, 64, 4, 32)],
)
def test_setup_cache(max_batch_size, max_seq_len, heads, head_dim):
    layers = [
        Mock(attention=Mock(kw_cache=None)),
        Mock(attention=Mock(kw_cache=None)),
    ]
    block_size = 64
    rope_base = 1000
    setup_cache(
        max_batch_size,
        max_seq_len,
        head_dim * heads,
        heads,
        layers,
        block_size,
        rope_base,
    )
    for layer in layers:
        assert isinstance(layer.attention.kw_cache, KVCache)


# 1. Edge Cases
def test_find_multiple_edge_cases():
    assert find_multiple(0, 5) == 0
    assert find_multiple(5, 0) == 5
    assert find_multiple(0, 0) == 0


def test_precompute_freq_cis_edge_cases():
    seq_len = 128
    n_elem = 0
    freqs = precompute_freq_cis(seq_len, n_elem)
    assert freqs.shape == torch.Size([seq_len, 0, 2])


# 2. Additional KVCache Tests
def test_kv_cache_update_empty_input():
    cache = KVCache(32, 128, 8, 64)
    input_pos = torch.tensor([], dtype=torch.int64)
    k_val = torch.randn(32, 8, 64, 64)
    v_val = torch.randn(32, 8, 64, 64)
    k_out, v_out = cache.update(input_pos, k_val, v_val)
    assert k_out.shape == torch.Size([32, 8, 128, 64])
    assert v_out.shape == torch.Size([32, 8, 128, 64])


def test_kv_cache_update_out_of_bounds_input():
    cache = KVCache(32, 128, 8, 64)
    input_pos = torch.tensor([140, 160, 200], dtype=torch.int64)
    k_val = torch.randn(32, 8, 64, 64)
    v_val = torch.randn(32, 8, 64, 64)
    k_out, v_out = cache.update(input_pos, k_val, v_val)
    assert k_out.shape == torch.Size([32, 8, 128, 64])
    assert v_out.shape == torch.Size([32, 8, 128, 64])


# 3. Additional setup_cache Tests
def test_setup_cache_max_seq_len_greater_than_max():
    layers = [
        Mock(attention=Mock(kw_cache=None)),
        Mock(attention=Mock(kw_cache=None)),
    ]
    max_batch_size = 16
    max_seq_len = 64
    heads = 4
    head_dim = 32
    block_size = 32
    rope_base = 1000
    setup_cache(
        max_batch_size,
        max_seq_len + 10,
        head_dim * heads,
        heads,
        layers,
        block_size,
        rope_base,
    )
    for layer in layers:
        assert isinstance(layer.attention.kw_cache, KVCache)
        assert layer.attention.kw_cache.k_cache.shape == torch.Size(
            [max_batch_size, heads, max_seq_len + 10, head_dim]
        )
        assert layer.attention.kw_cache.v_cache.shape == torch.Size(
            [max_batch_size, heads, max_seq_len + 10, head_dim]
        )


def test_setup_cache_max_batch_size_greater_than_max():
    layers = [
        Mock(attention=Mock(kw_cache=None)),
        Mock(attention=Mock(kw_cache=None)),
    ]
    max_batch_size = 64
    max_seq_len = 32
    heads = 4
    head_dim = 32
    block_size = 32
    rope_base = 1000
    setup_cache(
        max_batch_size + 10,
        max_seq_len,
        head_dim * heads,
        heads,
        layers,
        block_size,
        rope_base,
    )
    for layer in layers:
        assert isinstance(layer.attention.kw_cache, KVCache)
        assert layer.attention.kw_cache.k_cache.shape == torch.Size(
            [max_batch_size + 10, heads, max_seq_len, head_dim]
        )
        assert layer.attention.kw_cache.v_cache.shape == torch.Size(
            [max_batch_size + 10, heads, max_seq_len, head_dim]
        )
