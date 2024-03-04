import torch
from einops import pack, rearrange, repeat, unpack
from torch import einsum, nn

from zeta.nn.embeddings.sinusoidal import (
    SinusoidalEmbeddings,
    apply_rotary_pos_emb,
)
from zeta.utils.main import (
    default,
    exists,
    l2norm,
    look_around,
    max_neg_values,
    pad_to_multiple,
)

# constant
TOKEN_SELF_ATTN_VALUE = -5e4


class LocalAttention(nn.Module):
    """

    The LocalAttention module provides a mechanism to perform local attention operations.
    Unlike global attention where every token can attend to every other token,
    in local attention each token can only attend to a subset of tokens within a defined window. This reduces the computational cost and captures the local structure in sequences like text or time-series data.

    Args:
        window_size: (int) The size of the attention window.
        causal: (bool, optional) If set to True, ensures causal attention. Default: False.
        look_backward: (int, optional) How many positions to look backward from the current position. Default: 1.
        look_forward: (int, optional) How many positions to look forward from the current position. Default: None which implies 0 if causal is True.
        dropout: (float, optional) Dropout rate for attention weights. Default: 0.1.
        shared_qk: (bool, optional) If set to True, the query and key are the same. Useful for certain types of attention mechanisms. Default: False.
        rel_pos_emb_config: (Optional) Deprecated. Configuration for the relative positional embeddings.
        dim: (int, optional) Dimension of embeddings. Only needed if rel_pos_emb_config is not provided.
        autopad: (bool, optional) If set to True, sequence will be automatically padded to be divisible by the window size. Default: False.
        exact_windowsize: (bool, optional) Ensures exact window size for non-causal attention. Default: False.
        scale: (Optional) Scaling factor for the queries.
        use_rotary_pos_emb: (bool, optional) If set to True, rotary positional embeddings will be used. Default: True.
        use_xpos: (bool, optional) If set to True, allows for extrapolation of window sizes. Requires use_rotary_pos_emb to be True. Default: False.
        xpos_scale_base: (Optional) Base scaling factor for extrapolated window sizes.

    Usage:
    >>> model = LocalAttention(64, 1, 1, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape

    """

    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        dropout=0.1,
        shared_qk=False,
        rel_pos_emb_config=None,
        dim=None,
        autopad=False,
        exact_windowsize=False,
        scale=None,
        use_rotary_pos_emb=True,
        use_xpos=False,
        xpos_scale_base=None,
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (
            causal and look_forward > 0
        ), "you cannot look forward if causal"

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (
            exists(rel_pos_emb_config) or exists(dim)
        ):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos=use_xpos,
                scale_base=default(xpos_scale_base, window_size // 2),
            )

    """

    Forward Method
        Parameters
        q: (Tensor) The query tensor.

        k: (Tensor) The key tensor.

        v: (Tensor) The value tensor.

        mask: (Optional[Tensor]) A mask tensor for the keys. Can also be passed as input_mask.

        input_mask: (Optional[Tensor]) Another way to pass the mask tensor for keys.

        attn_bias: (Optional[Tensor]) Additional biases to add to the attention scores.

        window_size: (Optional[int]) If provided, this window size will override the default window size defined during initialization.

        Returns
        out: (Tensor) The output tensor after the attention operation.
    """

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        input_mask=None,
        attn_bias=None,
        window_size=None,
    ):
        mask = default(mask, input_mask)

        assert not (
            exists(window_size) and not self.use_xpos
        ), "cannot perform window size extrapolation if xpos is not turned on"

        (
            shape,
            autopad,
            pad_value,
            window_size,
            causal,
            look_backward,
            look_forward,
            shared_qk,
        ) = (
            q.shape,
            self.autopad,
            -1,
            default(window_size, self.window_size),
            self.causal,
            self.look_backward,
            self.look_forward,
            self.shared_qk,
        )

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(
            lambda t: pack([t], "* n d"), (q, k, v)
        )

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(
                lambda t: pad_to_multiple(t, self.window_size, dim=-2),
                (q, k, v),
            )

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head**-0.5)

        assert (n % window_size) == 0, (
            f"sequence length {n} must be divisible by window size"
            f" {window_size} for local attention"
        )

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device=device)
        b_t = rearrange(seq, "(w n) -> 1 w n", w=windows, n=window_size)

        # bucketing

        bq, bk, bv = map(
            lambda t: rearrange(t, "b (w n) d -> b w n d", w=windows), (q, k, v)
        )

        bq = bq * scale

        look_around_kwargs = dict(
            backward=look_backward, forward=look_forward, pad_value=pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale=xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, "... i -> ... i 1")
        bq_k = rearrange(bq_k, "... j -> ... 1 j")

        pad_mask = bq_k == pad_value

        sim = einsum("b h i e, b h j e -> b h i j", bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, "h i j -> (b h) 1 i j", b=b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_values(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = self.window_size * self.look_backward
                causal_mask = causal_mask | (
                    bq_t > (bq_k + max_causal_window_size)
                )

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = self.window_size * self.look_backward
            max_forward_window_size = self.window_size * self.look_forward
            window_mask = (
                ((bq_k - max_forward_window_size) > bq_t)
                | (bq_t > (bq_k + max_backward_window_size))
                | pad_mask
            )
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(
                    mask, window_size, dim=-1, value=False
                )

            mask = rearrange(
                mask, "... (w n) -> (...) w n", w=windows, n=window_size
            )
            mask = look_around(
                mask, **{**look_around_kwargs, "pad_value": False}
            )
            mask = rearrange(mask, "... j -> ... 1 j")
            mask = repeat(mask, "b ... -> (b h) ...", h=h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum("b h i j, b h j e -> b h i e", attn, bv)
        out = rearrange(out, "b w n d -> b (w n) d")

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, "* n d")
        return out
