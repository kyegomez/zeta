import math
import warnings
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from einops import rearrange
from packaging import version

from zeta.nn.attention.base import BaseAttention


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LPLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias)
            if self.bias is not None
            else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


def rms_norm(x, weight=None, eps=1e-5):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        weight=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)


class LPRMSNorm(RMSNorm):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        weight=True,
        dtype=None,
        device=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            weight=weight,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        with torch.autocast(enabled=False, device_type=x.device_type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(
                dtype=x.dtype
            )


# Registers
FC_CLASS_REGISTRY = {
    "torch": nn.Linear,
}

NORM_CLASS_REGISTRY = {
    "layernornm": nn.LayerNorm,
    "low_precision_layernorm": LPLayerNorm,
    "rmsnorm": LPLayerNorm,
    "low_precision_rmsnorm": LPRMSNorm,
}


def _reset_causal(
    num_query_tokens: int, num_key_tokens: int, original_causal: bool
):
    # disable causal when it is not needed
    # necessary for flash & triton for generation with kv_cache
    if original_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                "MPT does not support query and key with different number of"
                " tokens, unless number of query tokens is 1."
            )
        else:
            return False
    return original_causal


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    heads,
    past_key_value=None,
    softmax_scale=None,
    bias=None,
    key_padding_mask=None,
    causal=False,
    dropout=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    q = rearrange(query, "b s (h d) -> b h s d", h=heads)
    kv_heads = 1 if multiquery else heads
    k = rearrange(key, "b s (h d) -> b h d s", h=kv_heads)
    v = rearrange(value, "b s (h d) -> b h s d", h=kv_heads)

    if past_key_value is not None:
        # attn_impl: flash & triton use kernels which expect input shape [b, s, h, d_head].
        # kv_cache is therefore stored using that shape.
        # attn_impl: torch stores the kv_cache in the ordering which is most advantageous
        # for its attn computation ie
        # keys are stored as tensors with shape [b, h, d_head, s] and
        # values are stored as tensors with shape [b, h, s, d_head]
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, bias.size(2) - s_q)
        _s_k = max(0, bias.size(3) - s_k)
        bias = bias[:, :, _s_q:, _s_k:]

        if (bias.size(-1) != 1 and bias.size(-1) != s_k) or (
            bias.size(-2) != 1 and bias.size(-2) != s_q
        ):
            raise RuntimeError(
                f"bias (shape: {bias.shape}) is expected to broadcast to shape:"
                f" {attn_weight.shape}."
            )
        attn_weight = attn_weight + bias

    min_val = torch.finfo(q.dtype).min

    if key_padding_mask is not None:
        if bias is not None:
            warnings.warn(
                "Propogating key_padding_mask to the attention module "
                + "and applying it within the attention module can cause "
                + "unneccessary computation/memory usage. Consider integrating "
                + "into bias once and passing that to each attention "
                + "module instead."
            )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val
        )

    if causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(
            causal_mask.view(1, 1, s_q, s_k), min_val
        )

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout:
        attn_weight = torch.nn.functional.dropout(
            attn_weight, p=dropout, training=training, inplace=True
        )

    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, "b h s d -> b s (h d)")

    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f"{tensor.dtype=} must be in {valid_dtypes=}.")
        if not tensor.is_cuda:
            raise TypeError(f"Inputs must be cuda tensors ({tensor.is_cuda=}).")


def flash_attn_fn(
    query,
    key,
    value,
    heads,
    past_key_value=None,
    softmax_scale=None,
    bias=None,
    key_padding_mask=None,
    causal=False,
    dropout=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    try:
        # type: ignore # yapf: disable # isort: skip
        from flash_attn import bert_padding, flash_attn_interface
    except BaseException:
        raise RuntimeError("Please install flash-attn==1.0.3.post0")

    check_valid_inputs(query, key, value)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, bias.size(2) - query.size(1))
        _s_k = max(0, bias.size(3) - key.size(1))
        bias = bias[:, :, _s_q:, _s_k:]

    if bias is not None:
        raise NotImplementedError("bias not implemented for flash attn.")

    batch_size, seqlen = query.shape[:2]

    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1) :]

    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = (
        bert_padding.unpad_input(query, query_padding_mask)
    )
    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=heads)

    key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(
        key, key_padding_mask
    )
    key_unpad = rearrange(
        key_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else heads
    )

    value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(
        value_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else heads
    )

    if multiquery:
        key_unpad = key_unpad.expand(
            key_unpad.size(0), heads, key_unpad.size(-1)
        )
        value_unpad = value_unpad.expand(
            value_unpad.size(0), heads, value_unpad.size(-1)
        )

    dropout = dropout if training else 0.0

    reset_causal = _reset_causal(query.size(1), key.size(1), causal)

    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout,
        softmax_scale=softmax_scale,
        causal=reset_causal,
        return_attn_probs=needs_weights,
    )

    output = bert_padding.pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
        indices_q,
        batch_size,
        seqlen,
    )
    return output, None, past_key_value


def attn_bias_shape(
    attn_impl, heads, seq_len, alibi, prefix_lm, causal, use_sequence_id
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, heads, seq_len, seq_len)
            return (1, heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f"{attn_impl=} is an invalid setting.")


def build_attn_bias(
    attn_impl,
    bias,
    heads,
    seq_len,
    causal=False,
    alibi=False,
    alibi_bias_max=8,
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            # in place add alibi to attn bias
            device, dtype = bias.device, bias.dtype
            bias = bias.add(
                build_alibi_bias(
                    heads,
                    seq_len,
                    full=not causal,
                    alibi_bias_max=alibi_bias_max,
                    device=device,
                    dtype=dtype,
                )
            )
        return bias
    else:
        raise ValueError(f"{attn_impl=} is an invalid setting.")


# helper helpers
def gen_slopes(heads, alibi_bias_max=8, device=None):
    _heads = 2 ** math.ceil(math.log2(heads))
    m = torch.arange(1, _heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _heads)
    slopes = 1.0 / torch.pow(2, m)

    if _heads != heads:
        # if heads is not a power of two,
        # Huggingface and FasterTransformer calculate slopes normally,
        # then return this strided concatenation of slopes
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:heads]

    return slopes.view(1, heads, 1, 1)


def build_alibi_bias(
    heads,
    seq_len,
    full=False,
    alibi_bias_max=8,
    device=None,
    dtype=None,
):
    alibi_bias = torch.arange(
        1 - seq_len, 1, dtype=torch.int32, device=device
    ).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to
        # the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=torch.int32, device=device
        ).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)


def triton_flash_attn_fn(
    query,
    key,
    value,
    heads,
    past_key_value=None,
    softmax_scale=None,
    bias=None,
    key_padding_mask=None,
    causal=False,
    dropout=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    try:
        from llmfoundry.models.layers.flash_attn_triton import flash_attn_func
    except BaseException:
        _installed = False
        if version.parse(torch.__version__) < version.parse("2.0.0"):
            _installed = True
            # if torch1.13.1 revert to using triton flash attn from HazyResearch
            # with flash-attn==1.0.3.post0 and triton==2.0.0.dev20221202
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except BaseException:
                _installed = False
        if not _installed:
            # installing triton-pre-mlir works for both torch1.13.1 and torch2.0+
            # default recommendation is to install this variant
            raise RuntimeError(
                "Requirements for `attn_impl: triton` not installed. Either (1)"
                " have a CUDA-compatible GPU and `pip install .[gpu]` if"
                " installing from source or `pip install"
                " triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python`"
                " if installing from pypi, or (2) use torch attn"
                " model.attn_config.attn_impl=torch (torch attn_impl will be"
                " slow). Note: (1) requires you have CMake and PyTorch already"
                " installed."
            )

    check_valid_inputs(query, key, value)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, bias.size(2) - query.size(1))
        _s_k = max(0, bias.size(3) - key.size(1))
        bias = bias[:, :, _s_q:, _s_k:]

    if dropout:
        raise NotImplementedError(
            "Dropout not implemented for attn_impl: triton."
        )

    if needs_weights:
        raise NotImplementedError(
            "attn_impl: triton cannot return attn weights."
        )

    if key_padding_mask is not None:
        warnings.warn(
            "Propagating key_padding_mask to the attention module "
            + "and applying it within the attention module can cause "
            + "unnecessary computation/memory usage. Consider integrating "
            + "into bias once and passing that to each attention "
            + "module instead."
        )
        b_size, s_k = key_padding_mask.shape[:2]

        if bias is None:
            bias = query.new_zeros(b_size, 1, 1, s_k)

        bias = bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)),
            torch.finfo(query.dtype).min,
        )

    query = rearrange(query, "b s (h d) -> b s h d", h=heads)
    key = rearrange(key, "b s (h d) -> b s h d", h=1 if multiquery else heads)
    value = rearrange(
        value, "b s (h d) -> b s h d", h=1 if multiquery else heads
    )

    if multiquery:
        # necessary to repeat instead of expand tensor because
        # output contains NaN in edge cases such as with head dimension = 8
        key = key.repeat(1, 1, heads, 1)
        value = value.repeat(1, 1, heads, 1)

    reset_causal = _reset_causal(query.size(1), key.size(1), causal)
    attn_output = flash_attn_func(
        query, key, value, bias, reset_causal, softmax_scale
    )

    output = attn_output.view(*attn_output.shape[:2], -1)

    return output, None, past_key_value


class MultiHeadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        attn_impl: str = "triton",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        fc_type: str = "torch",
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.heads = heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.heads)
        self.attn_dropout = attn_pdrop

        fc_kwargs = {}
        if fc_type != "te":
            fc_kwargs["device"] = device
        self.Wqkv = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            3 * self.d_model,
            **fc_kwargs,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
            self.q_ln = norm_class(self.d_model, device=device)
            self.k_ln = norm_class(self.d_model, device=device)

        if self.attn_impl == "flash":
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == "triton":
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    "While `attn_impl: triton` can be faster than `attn_impl:"
                    " flash` "
                    + "it uses more memory. When training larger models"
                    " this can"
                    " trigger "
                    + "alloc retries which hurts performance. If"
                    " encountered, we"
                    " recommend "
                    + "using `attn_impl: flash` if your model does not use"
                    " `alibi` or `prefix_lm`."
                )
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    "Using `attn_impl: torch`. If your model does not use"
                    " `alibi` or "
                    + "`prefix_lm` we recommend using `attn_impl: flash`"
                    " otherwise "
                    + "we recommend using `attn_impl: triton`."
                )
        else:
            raise ValueError(f"{attn_impl=} is an invalid setting.")

        self.out_proj = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            self.d_model,
            **fc_kwargs,
        )
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        bias=None,
        mask=None,
        causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)

        key_padding_mask = mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            bias=bias,
            key_padding_mask=key_padding_mask,
            causal=causal,
            dropout=self.attn_dropout,
            training=self.training,
            needs_weights=needs_weights,
        )

        return self.out_proj(context), attn_weights, past_key_value


class MultiQueryAttention(BaseAttention):
    """Multi-Query self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.

    Look for documentation
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        attn_impl: str = "torch",
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        fc_type: str = "torch",
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout = attn_pdrop

        fc_kwargs = {}
        if fc_type != "te":
            fc_kwargs["device"] = device
        # - vchiley
        self.Wqkv = FC_CLASS_REGISTRY[fc_type](
            d_model,
            d_model + 2 * self.head_dim,
            **fc_kwargs,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
            self.q_ln = norm_class(d_model, device=device)
            self.k_ln = norm_class(self.head_dim, device=device)

        if self.attn_impl == "flash":
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == "triton":
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    "While `attn_impl: triton` can be faster than `attn_impl:"
                    " flash` "
                    + "it uses more memory. When training larger models"
                    " this can"
                    " trigger "
                    + "alloc retries which hurts performance. If"
                    " encountered, we"
                    " recommend "
                    + "using `attn_impl: flash` if your model does not use"
                    " `alibi` or `prefix_lm`."
                )
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    "Using `attn_impl: torch`. If your model does not use"
                    " `alibi` or "
                    + "`prefix_lm` we recommend using `attn_impl: flash`"
                    " otherwise "
                    + "we recommend using `attn_impl: triton`."
                )
        else:
            raise ValueError(f"{attn_impl=} is an invalid setting.")

        self.out_proj = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            self.d_model,
            **fc_kwargs,
        )
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        bias=None,
        mask=None,
        causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2
        )

        key_padding_mask = mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            bias=bias,
            key_padding_mask=key_padding_mask,
            causal=causal,
            dropout=self.attn_dropout,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )

        return self.out_proj(context), attn_weights, past_key_value
