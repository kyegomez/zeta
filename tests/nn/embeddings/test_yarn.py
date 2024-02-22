import pytest
import torch

from zeta.nn.embeddings.yarn import YarnEmbedding


def test_yarnembedding_initialization():
    model = YarnEmbedding(dim=512)
    assert isinstance(model, YarnEmbedding)
    assert model.dim == 512
    assert model.max_position_embeddings == 2048
    assert model.base == 10000


def test_yarnembedding_forward():
    model = YarnEmbedding(dim=512)
    x = torch.randn(1, 10, 512)
    cos_cached, sin_cached = model(x, seq_len=10)
    assert cos_cached.shape == (1, 1, 10, 512)
    assert sin_cached.shape == (1, 1, 10, 512)


@pytest.mark.parametrize("seq_len", [0])
def test_yarnembedding_forward_edge_cases(seq_len):
    model = YarnEmbedding(dim=512)
    x = torch.randn(1, seq_len, 512)
    with pytest.raises(Exception):
        model(x, seq_len=seq_len)


def test_yarnembedding_forward_invalid_dimensions():
    model = YarnEmbedding(dim=512)
    x = torch.randn(1, 10, 256)
    with pytest.raises(Exception):
        model(x, seq_len=10)


# Test case for default initialization
def test_default_init():
    dim = 10
    module = YarnEmbedding(dim)
    assert module.dim == dim
    assert module.max_position_embeddings == 2048
    assert module.base == 10000
    assert module.original_max_position_embeddings == 2048
    assert module.extrapolation_factor == 1
    assert module.attn_factor == 1
    assert module.beta_fast == 32
    assert module.beta_slow == 1
    assert not module.finetuned
    assert module.device is None
    assert isinstance(module.inv_freq, torch.Tensor)
    assert module.mscale == 1
    assert module.max_seq_len_cached == 2048
    assert isinstance(module.cos_cached, torch.Tensor)
    assert isinstance(module.sin_cached, torch.Tensor)


# Test case for finetuned initialization
def test_finetuned_init():
    dim = 10
    module = YarnEmbedding(dim, finetuned=True)
    assert module.dim == dim
    assert module.max_position_embeddings == 2048
    assert module.base == 10000
    assert module.original_max_position_embeddings == 2048
    assert module.extrapolation_factor == 1
    assert module.attn_factor == 1
    assert module.beta_fast == 32
    assert module.beta_slow == 1
    assert module.finetuned
    assert module.device is None
    assert isinstance(module.inv_freq, torch.Tensor)
    assert module.mscale == 1
    assert module.max_seq_len_cached == 2048
    assert isinstance(module.cos_cached, torch.Tensor)
    assert isinstance(module.sin_cached, torch.Tensor)


# Test case for forward pass with default parameters
def test_forward_pass_default_params():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(10, 10)
    cos_emb, sin_emb = module(x, seq_len=10)
    assert cos_emb.shape == (1, 1, 10, 10)
    assert sin_emb.shape == (1, 1, 10, 10)


# Test case for forward pass with custom sequence length
def test_forward_pass_custom_seq_len():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(10, 10)
    cos_emb, sin_emb = module(x, seq_len=5)
    assert cos_emb.shape == (1, 1, 5, 10)
    assert sin_emb.shape == (1, 1, 5, 10)


# Test case for forward pass with larger sequence length than cached
def test_forward_pass_larger_seq_len():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(10, 10)
    cos_emb, sin_emb = module(x, seq_len=4096)
    assert cos_emb.shape == (1, 1, 4096, 10)
    assert sin_emb.shape == (1, 1, 4096, 10)


# Test case for yarn method
def test_yarn_method():
    dim = 10
    module = YarnEmbedding(dim)
    module.yarn(0.5, device=torch.device("cpu"))
    assert isinstance(module.inv_freq, torch.Tensor)
    assert module.mscale == 1


# Test case for custom initialization
def test_custom_init():
    dim = 10
    max_position_embeddings = 4096
    base = 5000
    original_max_position_embeddings = 2048
    extrapolation_factor = 2
    attn_factor = 2
    beta_fast = 16
    beta_slow = 2
    finetuned = True
    device = torch.device("cuda")
    module = YarnEmbedding(
        dim,
        max_position_embeddings,
        base,
        original_max_position_embeddings,
        extrapolation_factor,
        attn_factor,
        beta_fast,
        beta_slow,
        finetuned,
        device,
    )
    assert module.dim == dim
    assert module.max_position_embeddings == max_position_embeddings
    assert module.base == base
    assert (
        module.original_max_position_embeddings
        == original_max_position_embeddings
    )
    assert module.extrapolation_factor == extrapolation_factor
    assert module.attn_factor == attn_factor
    assert module.beta_fast == beta_fast
    assert module.beta_slow == beta_slow
    assert module.finetuned == finetuned
    assert module.device == device


# Test case for forward pass with default values
def test_forward_pass_default_values():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(10, 10)
    seq_len = 10
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.shape == (1, 1, seq_len, dim // 2)
    assert sin_embed.shape == (1, 1, seq_len, dim // 2)


# Test case for forward pass with custom values
def test_forward_pass_custom_values():
    dim = 10
    max_position_embeddings = 32
    base = 5000
    original_max_position_embeddings = 16
    extrapolation_factor = 2
    attn_factor = 2
    beta_fast = 16
    beta_slow = 2
    finetuned = True
    device = torch.device("cuda")
    module = YarnEmbedding(
        dim,
        max_position_embeddings,
        base,
        original_max_position_embeddings,
        extrapolation_factor,
        attn_factor,
        beta_fast,
        beta_slow,
        finetuned,
        device,
    )
    x = torch.randn(1, 1, 10, dim)
    seq_len = 10
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.shape == (1, 1, seq_len, dim // 2)
    assert sin_embed.shape == (1, 1, seq_len, dim // 2)


# Test case for forward pass with a larger sequence length
def test_forward_pass_large_seq_len():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(1, 1, 20, dim)
    seq_len = 20
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.shape == (1, 1, seq_len, dim // 2)
    assert sin_embed.shape == (1, 1, seq_len, dim // 2)


# Test case for forward pass with finetuned embeddings
def test_forward_pass_finetuned():
    dim = 10
    max_position_embeddings = 16
    base = 5000
    original_max_position_embeddings = 8
    extrapolation_factor = 2
    attn_factor = 2
    beta_fast = 16
    beta_slow = 2
    finetuned = True
    device = torch.device("cuda")
    module = YarnEmbedding(
        dim,
        max_position_embeddings,
        base,
        original_max_position_embeddings,
        extrapolation_factor,
        attn_factor,
        beta_fast,
        beta_slow,
        finetuned,
        device,
    )
    x = torch.randn(1, 1, 5, dim)
    seq_len = 5
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.shape == (1, 1, seq_len, dim // 2)
    assert sin_embed.shape == (1, 1, seq_len, dim // 2)


# Test case for forward pass with a different device
def test_forward_pass_different_device():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(1, 1, 5, dim)
    seq_len = 5
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.device == torch.device("cpu")
    assert sin_embed.device == torch.device("cpu")


# Test case for forward pass with a different device (GPU)
def test_forward_pass_gpu_device():
    dim = 10
    device = torch.device("cuda")
    module = YarnEmbedding(dim, device=device)
    x = torch.randn(1, 1, 5, dim, device=device)
    seq_len = 5
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.device == device
    assert sin_embed.device == device


# Test case for updating the embeddings when sequence length increases
def test_update_embeddings_on_sequence_length_increase():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(1, 1, 20, dim)
    seq_len = 20
    cos_embed_before, sin_embed_before = module(x, seq_len)

    # Increase sequence length
    x = torch.randn(1, 1, 30, dim)
    seq_len = 30
    cos_embed_after, sin_embed_after = module(x, seq_len)

    assert cos_embed_before.shape != cos_embed_after.shape
    assert sin_embed_before.shape != sin_embed_after.shape


# Test case for updating the embeddings when sequence length decreases
def test_update_embeddings_on_sequence_length_decrease():
    dim = 10
    module = YarnEmbedding(dim)
    x = torch.randn(1, 1, 30, dim)
    seq_len = 30
    cos_embed_before, sin_embed_before = module(x, seq_len)

    # Decrease sequence length
    x = torch.randn(1, 1, 20, dim)
    seq_len = 20
    cos_embed_after, sin_embed_after = module(x, seq_len)

    assert cos_embed_before.shape != cos_embed_after.shape
    assert sin_embed_before.shape != sin_embed_after.shape


# Test case for forward pass with GPU device
@pytest.mark.gpu
def test_forward_pass_gpu():
    dim = 10
    module = YarnEmbedding(dim, device=torch.device("cuda"))
    x = torch.randn(1, 1, 10, dim).to(torch.device("cuda"))
    seq_len = 10
    cos_embed, sin_embed = module(x, seq_len)
    assert cos_embed.device == torch.device("cuda")
    assert sin_embed.device == torch.device("cuda")
