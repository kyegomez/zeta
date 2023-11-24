from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from zeta.nn.attention.base import BaseAttention
from zeta.nn.attention.flash_attention import FlashAttention
from zeta.nn.biases.relative_position_bias import RelativePositionBias
from zeta.nn.embeddings.xpos_relative_position import XPOS

device = "cuda:0"
dtype = torch.float16


class ParallelWrapper:
    """
    A simple wrapper to enable easy usage of data parallelism.

    Arguments:
        model: The neural network model to be parallelized.
        device (optional): The device to which the model should be moved. Default: "cuda".
        use_data_parallel (optional): A boolean flag to indicate whether to use data parallelism or not. Default: True.
    """

    def __init__(self, model, device="cuda", use_data_parallel=True):
        self.model = model.to(device)
        self.use_data_parallel = use_data_parallel
        self.device = device

        if self.use_data_parallel and torch.cuda.device_count() < 1:
            print(f"Using {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __getattr__(self, name):
        # redirect attribute access to the internal model to allow direct
        # access to its methods and props
        return getattr(self.model, name)


# add alibi, qk layer norm, one write head, multihway,
class DilatedAttention(BaseAttention):
    """
    Dilated Attention Module.

    Arguments:
        d_model: The dimension of the attention layers.
        num_heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        casual (optional): If set to True, the attention mechanism is casual. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """

    def __init__(
        self,
        d_model: int = None,
        num_heads: int = None,
        dilation_rate: int = None,
        segment_size: int = None,
        dropout: int = 0.0,
        casual: bool = False,
        use_xpos: bool = False,
        use_rel_pos_bias: bool = False,
    ):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.dropout = nn.Dropout(dropout)
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        self.attention = FlashAttention(causal=self.casual, dropout=dropout).to(
            device
        )

        if use_xpos:
            self.xpos = XPOS(head_dim=d_model // num_heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(
                num_buckets=32, max_distance=128, n_heads=num_heads
            )

        # head offsets
        self.head_offsets = nn.Parameter(torch.randn(num_heads, d_model))

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(
            j - i + 2
        )

    def forward(self, x):
        print(f"X original shape: {x.shape} and x dtype: {x.dtype}")

        batch_size, seq_len, _ = x.shape
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0, 0, 0, padding_len))
        seq_len = seq_len + padding_len
        print(f"Paddex x shape:  {x.shape}")

        if self.use_xpos:
            x = self.xpos(x)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        print(f"z after view shape: {x.shape}")

        x = x[:, :, :: self.dilation_rate, :]
        print(f"x after dilation shape: {x.shape} and x.dtype: {x.dtype}")

        # Perform attention
        attn_output = self.attention(x, x, x)
        print(
            f"Attn output: {attn_output.shape} and dtype: {attn_output.dtype}"
        )

        # if use rel pos => apply relative positioning bias
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(
                batch_size, attn_output.size(1), attn_output.size(1)
            )
            print(
                f"attn_output: {attn_output.shape} and attn output:"
                f" {attn_output.dtype}"
            )

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))
            print(f"mask shape: {mask.shape} and mask dtype: {x.dtype}")

            attn_output = attn_output.masked_fill(mask, float("-inf"))
            print(
                f"attn output shape: {attn_output.shape} and attn_output:"
                f" {attn_output.dtype}"
            )

        # apply dropout
        attn_output = self.dropout(attn_output)
        print(
            f"attn output after dropout: {attn_output.shape} and dtype:"
            f" {attn_output.dtype}"
        )

        # Scatter and concatenate
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)
        print(
            f"attn_output scatter and concatenate: {attn_output.shape} and"
            f" {attn_output.dtype}"
        )
        return attn_output


class MultiheadDilatedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if not embed_dim % self.num_heads == 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads"
                f" ({num_heads})"
            )
        num_dilations = len(dilation_rates)
        num_segments = len(segment_lengths)
        if num_dilations != num_segments:
            raise ValueError(
                f"len(dilation_rates) ({num_dilations}) must be equal to "
                f"len(segment_lengths) ({num_segments})"
            )
        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be"
                " divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            # op=op,
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> Tuple[Tensor, None]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = self.attention(q, k, v, is_causal=is_causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)

        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, None
