import torch
from torch import nn, Tensor


# Helpers
def find_multiple(n: int, k: int) -> int:
    """Finds the smallest multiple of k that is greater than or equal to n.

    Args:
        n (int): _description_
        k (int): _description_

    Returns:
        int: _description_
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freq_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    """Precomputes the frequency values for the positional encodings.

    Args:
        seq_len (int): _description_
        n_elem (int): _description_
        base (int, optional): _description_. Defaults to 10000.

    Returns:
        Tensor: _description_
    """
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


class KVCache(nn.Module):
    """
    KVCache is a module that stores the key and value tensors for each
    position in the input sequence. This is used in the decoder of the
    Transformer model to store the key and value tensors for each position
    in the encoder output sequence.

    The cache is updated by calling the update method, which takes the
    input positions and the key and value tensors for those positions.

    The cache is a tensor of shape [B, H, S, D], where B is the batch size,
    H is the number of heads, S is the maximum sequence length, and D is
    the head dimension.

    Args:
        max_batch_size: The maximum batch size of the model.
        max_seq_len: The maximum sequence length of the model.
        heads: The number of heads in the model.
        head_dim: The dimension of each head.
        dtype: The datatype of the cache.

    Attributes:
        k_cache: The key cache.
        v_cache: The value cache.

    Methods:
        update: Updates the cache with the given input positions and key
            and value tensors.

    Input Shapes:
        input_pos: [S]
        k_val: [B, H, S, D]
        v_val: [B, H, S, D]

    Output Shapes:
        k_out: [B, H, S, D]
        v_out: [B, H, S, D]

    Examples:
    >>> from zeta.nn import KVCache
    >>> cache = KVCache(32, 128, 8, 64)
    >>> k_val = torch.randn(32, 8, 128, 64)
    >>> v_val = torch.randn(32, 8, 128, 64)
    >>> input_pos = torch.randint(0, 128, (5,))
    >>> k_out, v_out = cache.update(input_pos, k_val, v_val)
    >>> k_out.shape
    torch.Size([32, 8, 128, 64])
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        cache_shape = (max_batch_size, heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        """
        Updates the cache with the given input positions and key and value.

        Args:
            input_pos (_type_): _description_
            k_val (_type_): _description_
            v_val (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Input pos: [5], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos, :] = k_val
        v_out[:, :, input_pos, :] = v_val

        return k_out, v_out


def setup_cache(
    max_batch_size, max_seq_len, dim, heads, layers, block_size, rope_base
):
    """Sets up the cache for the given model.

    Args:
        max_batch_size (_type_): _description_
        max_seq_len (_type_): _description_
        dim (_type_): _description_
        heads (_type_): _description_
        layers (_type_): _description_
        block_size (_type_): _description_
        rope_base (_type_): _description_
    """
    if max_seq_len >= max_seq_len and max_batch_size >= max_batch_size:
        return

    head_dim = dim // heads
    max_seq_len = find_multiple(max_seq_len, 8)

    for b in layers:
        b.attention.kv_cache = KVCache(
            max_batch_size, max_seq_len, heads, head_dim
        )

    freq_cis = precompute_freq_cis(block_size, dim // heads, rope_base)
    causal_mask = torch.tril(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
    )

    return causal_mask, freq_cis
