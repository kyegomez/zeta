# prompts to jquesnelle
# https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaDynamicYaRNScaledRotaryEmbedding.py
import torch
from torch import nn
import math


# helpers
# inveerse dim formula to find dim based on number of rotations
def find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


# find dim range bounds based on rotations
def find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YarnEmbedding(nn.Module):
    """
    Yarn Embeddings.

    Args:
        dim (int): The dimension of the embeddings.
        max_position_embeddings (int): The maximum position embeddings.
        base (int): The base for the positional embeddings.
        original_max_position_embeddings (int): The original maximum position embeddings.
        extrapolation_factor (int): The extrapolation factor.
        attn_factor (int): The attention factor.
        beta_fast (int): The fast beta.
        beta_slow (int): The slow beta.
        finetuned (bool): Whether to finetune or not.
        device (torch.device): The device.


    Attributes:
        dim (int): The dimension of the embeddings.
        max_position_embeddings (int): The maximum position embeddings.
        base (int): The base for the positional embeddings.
        original_max_position_embeddings (int): The original maximum position embeddings.
        extrapolation_factor (int): The extrapolation factor.
        attn_factor (int): The attention factor.
        beta_fast (int): The fast beta.
        beta_slow (int): The slow beta.
        finetuned (bool): Whether to finetune or not.
        device (torch.device): The device.
        inv_freq (torch.Tensor): The inverse frequencies.
        mscale (float): The mscale.
        max_seq_len_cached (int): The maximum sequence length cached.
        cos_cached (torch.Tensor): The cached cosine.
        sin_cached (torch.Tensor): The cached sine.


    Example:
        >>> module = YarnEmbedding(10)
        >>> x = torch.randn(10, 10)
        >>> y = module(x)
        >>> y.shape
        torch.Size([10, 10, 10])



    """

    def __init__(
        self,
        dim,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        original_max_position_embeddings: int = 2048,
        extrapolation_factor: int = 1,
        attn_factor: int = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        finetuned=False,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor

        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        if finetuned:
            self.yarn(
                self.max_position_embedding
                / self.original_max_position_embeddings,
                device,
            )
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2).float().to(device) / dim)
            )
            self.register_buffer("inv_freq", inv_freq)
            self.mscale = 1

        # build
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer(
            "cos_cached",
            (emb.cos() * self.mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            (emb.sin() * self.mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )

    def forward(self, x, seq_len=None):
        """forward"""
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            self.yarn(seq_len / self.original_max_position_embeddings, x.device)

            t = torch.arange(
                self.max_seq_len_cached,
                device=x.dtype,
                dtype=self.inv_freq.dtype,
            )

            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer(
                "cos_cached",
                (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype),
                persistent=False,
            )
            self.register_buffer(
                "sin_cached",
                (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype),
                persistent=False,
            )

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def yarn(self, scale, device):
        """Yarn Embeddings."""
        pos_freqs = self.base ** (
            torch.arange(0, self.dim, 2).float().to(device) / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(get_mscale(scale) * self.attn_factor)
