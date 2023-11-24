from collections import namedtuple
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import Tensor, einsum, nn

from zeta.nn.attention.base import BaseAttention

# constants

EfficientAttentionConfig = namedtuple(
    "EfficientAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)

# helpers


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


@dataclass
class Intermediates:
    """
    Dataclass to store intermediate tensors during attention computation.

    Args:
        qk_similarities (torch.Tensor): Tensor storing the similarities between query and key.
        pre_softmax_attn (torch.Tensor): Tensor storing the attention weights before softmax.
        post_softmax_attn (torch.Tensor): Tensor storing the attention weights after softmax.

    Methods:
        to_tuple(): Convert the Intermediates object to a tuple.

    """

    qk_similarities: Tensor = None
    pre_softmax_attn: Tensor = None
    post_softmax_attn: Tensor = None

    def to_tuple(self):
        """
        Convert the Intermediates object to a tuple.

        Returns:
            tuple: Tuple representation of the Intermediates object.
        """
        return (
            self.qk_similarities,
            self.pre_softmax_attn,
            self.post_softmax_attn,
        )


class FlashAttention(BaseAttention):
    def __init__(
        self, causal: bool = False, dropout: float = 0.0, flash: bool = True
    ):
        """
        FlashAttention module that performs attention computation.

        Args:
            causal (bool): Whether to apply causal masking (default: False).
            dropout (float): Dropout probability (default: 0.).
            flash (bool): Whether to use flash attention (default: True).

        """
        super().__init__()

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), (
            "in order to use flash attention, you must be using pytorch 2.0 or"
            " above"
        )

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda")
        )

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on"
                " cuda"
            )
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention"
                " if input tensor is on cuda"
            )
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, i, j, device):
        """
        Generate a mask for attention computation.

        Args:
            i (int): Length of the query sequence.
            j (int): Length of the key sequence.
            device (torch.device): Device to place the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (i, j).

        """
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(
            j - i + 1
        )

    def flash_attn(self, q, k, v, mask=None, attn_bias=None):
        """
        Perform flash attention computation.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, heads, q_len, dim).
            k (torch.Tensor): Key tensor of shape (batch, heads, k_len, dim).
            v (torch.Tensor): Value tensor of shape (batch, heads, v_len, dim).
            mask (torch.Tensor): Mask tensor of shape (batch, heads, q_len, k_len) (default: None).
            attn_bias (torch.Tensor): Attention bias tensor of shape (batch, heads, q_len, k_len) (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch, heads, q_len, dim).

        """
        batch, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention
        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = self.create_causal_mask(
                    q_len, k_len, device=device
                )
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "h i j -> 1 h i j").expand(
                batch, heads, -1, -1
            )

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi
            # positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(
                    q_len, k_len, device=device
                )
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )

            return out

    def forward(self, q, k, v, mask=None, attn_bias=None):
        """
        Perform attention computation.

        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension

        Args:
            q (torch.Tensor): Query tensor of shape (batch, heads, q_len, dim).
            k (torch.Tensor): Key tensor of shape (batch, heads, k_len, dim).
            v (torch.Tensor): Value tensor of shape (batch, heads, v_len, dim).
            mask (torch.Tensor): Mask tensor of shape (batch, heads, q_len, k_len) (default: None).
            attn_bias (torch.Tensor): Attention bias tensor of shape (batch, heads, q_len, k_len) (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch, heads, q_len, dim).

        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
