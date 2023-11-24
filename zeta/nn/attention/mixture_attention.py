import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from typing import Tuple, Optional
from einops import rearrange, repeat, reduce
from zeta.models.vit import exists
from zeta.structs.transformer import RMSNorm, apply_rotary_pos_emb

from zeta.nn.attention.attend import Attend
from zeta.nn.attention.local_attention_mha import LocalMHA
from zeta.utils.main import default, pad_to_multiple

from colt5_attention import CoordinateDescentRouter


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        dim_context=None,
        heads=8,
        causal=False,
        groups=1,
        dropout=0.0,
        flash=False,
        prenorm=False,
    ):
        super().__init__()
        self.heads = heads
        self.groups = groups

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)
        self.norm = RMSNorm(dim, groups=groups) if prenorm else nn.Identity()

        self.context_norm = (
            RMSNorm(dim_context, groups=groups) if prenorm else nn.Identity()
        )

        self.attend = Attend(dropout=dropout, causal=causal, flash=flash)

        # null key value to proetect againsta  row that is masked
        self.null_kv = nn.Parameter(torch.randn(2, groups, heads, 1, dim_head))

        # utilizing convo groups to process experts in parallel
        self.to_q = nn.Conv1d(
            dim * groups, dim_inner * groups, 1, bias=False, groups=groups
        )
        self.to_kv = nn.Conv1d(
            dim_context * groups,
            dim_inner * 2 * groups,
            1,
            bias=False,
            groups=groups,
        )
        self.to_out = nn.Conv1d(
            dim_inner * groups, dim * groups, 1, bias=False, groups=groups
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        queries_scale=None,
        keys_scale=None,
        values_scale=None,
        output_scale=None,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Einops
        b - batch
        g - groups
        n - sequence
        d - feature dimension
        """

        b, g, h = x.shape[0], self.groups, self.heads
        one_expert = x.ndim == 3

        if one_expert:
            assert g == 1
            x = rearrange(x, "b n d -> b 1 n d")

        assert x.ndim == 4
        assert x.shape[1] == g

        # fold the groups into the feature dimension to be processed in one go
        # by grouped convolutional
        x = rearrange(x, "b g n d -> b g d n")

        # handle context for cross attention
        if exists(context):
            context_one_expert = context.ndim == 3

            if context_one_expert:
                assert g == 1
                context = rearrange(context, "b n d -> b 1 n d")

            assert context.ndim == 4
            assert context.shape[1] == g

            context = rearrange(context, "b g n d -> b g d n")

        context = default(context, x)

        # take care of mask
        if exists(mask):
            if mask.ndim == 2:
                mask = repeat(mask, "b n -> (b g) n", g=g)
            elif mask.ndim == 3:
                mask = rearrange(mask, "b g n -> (b d) n")
            mask = F.pad(mask, (1, 0), value=True)

        # prenorm
        x = self.norm(x)
        context = self.context_norm(context)

        # fold groups into dimension for grouped conv
        x, context = map(
            lambda t: rearrange(t, "b g d n -> b (g d) n"), (x, context)
        )

        # q, k, v
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=1))

        # split heads and merge groups into batches
        q, k, v = map(
            lambda t: rearrange(t, "b (g h d) n -> b g h n d", h=h, g=g),
            (q, k, v),
        )

        # rotary embedding
        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb

            if q_rotary_emb.ndim > 2:
                q_rotary_emb = rearrange(q_rotary_emb, "b g n d -> b g 1 n d")

            if k_rotary_emb.ndim > 2:
                k_rotary_emb = rearrange(k_rotary_emb, "b g n d -> b g 1 n d")

            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)

        # give gradients to routed keys/values via normalized scores from the
        # router, if passed in
        if exists(queries_scale):
            q = q * queries_scale

        if exists(keys_scale):
            k = k * keys_scale

        if exists(values_scale):
            v = v * values_scale

        # merge group into batch
        q, k, v = map(lambda t: rearrange(t, "b g ... -> (b g) ..."), (q, k, v))

        # concat null key /values, to protect against a row having all masked
        # out elements and have a save a lot of headache
        nk, nv = map(
            lambda t: repeat(t, "g h 1 d -> (b g) h 1 d", b=b), self.null_kv
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # attention
        out = self.attend(q, k, v, mask=mask)

        # combine heads out
        out = rearrange(out, "(b g) h n d -> b (g h d) n", g=g)
        out = self.to_out(out)
        out = rearrange(out, "b (g d) n -> b g n d", g=g)

        if one_expert:
            out = rearrange(out, "b 1 n d -> b n d")

        if exists(output_scale):
            out = out * output_scale

        return out


class MixtureOfAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_routed_queries,
        num_routed_key_values,
        dim_context=None,
        local_attn=False,
        local_attn_window_size=None,
        num_experts,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_triton=True,
        flash_attn=True,
        prenorm=True,
        average_routed=False,
        **kwargs,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)
        self.num_routed_queries = num_routed_queries
        self.num_routed_key_values = num_routed_key_values

        self.null_routed_token = (
            nn.Parameter(torch.randn(1, 1, dim)) if not local_attn else None
        )

        self.average_routed = average_routed
        self.local_attn = None

        if local_attn:
            assert exists(local_attn_window_size)
            self.local_attn = LocalMHA(
                dim=dim,
                dim_head=dim_head,
                head=heads,
                prenorm=prenorm,
                window_size=local_attn_window_size,
            )

        self.query_router = CoordinateDescentRouter(
            dim, num_routing_tokens=num_experts, use_triton=use_triton, **kwargs
        )
        self.key_value_router = CoordinateDescentRouter(
            dim_context,
            num_routing_tokens=num_experts,
            use_triton=use_triton,
            **kwargs,
        )

        self.attn = Attention(
            dim=dim,
            dim_context=dim_context,
            dim_head=dim_head,
            heads=heads,
            groups=num_experts,
            dropout=dropout,
            flash=flash_attn,
            prenorm=prenorm,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        num_routed_queries=None,
        num_routed_key_values=None,
        rotary_emb=None,
    ):
        num_routed_queries = default(
            num_routed_queries, self.num_routed_queries
        )
        num_routed_key_values = default(
            num_routed_key_values, self.num_routed_key_values
        )

        is_cross_attn = exists(context)
        assert not (
            exists(self.local_attn) and is_cross_attn
        ), "cannot do cross attention with local attention"

        if not is_cross_attn:
            # self attention if context and context mask not passed in
            context = x
            context_mask = mask

        query_indices, query_scores, queries, query_mask = self.query_router(
            x, mask=mask, num_routed=num_routed_queries, keep_one_route_dim=True
        )
        rearrange(query_scores, "b g n -> b g n 1")

        (
            kv_indices,
            key_value_scores,
            key_values,
            key_value_mask,
        ) = self.key_value_router(
            context,
            mask=context_mask,
            num_tokens=num_routed_key_values,
            keep_one_route_dim=True,
        )
        key_value_scores = rearrange(key_value_scores, "b g n -> b g 1 n 1")

        # rotary embeddings
        if exists(rotary_emb):
            assert (
                not is_cross_attn
            ), "rotary embedding should not be used for cross attending"
            q_rotary_emb = (
                rotary_emb[query_indices]
                if exists(query_indices)
                else rotary_emb
            )
            k_rotary_emb = (
                rotary_emb[kv_indices] if exists(kv_indices) else rotary_emb
            )
            rotary_emb = (q_rotary_emb, k_rotary_emb)

        # attend
        attn_out = self.attn(
            queries,
            rotary_emb=rotary_emb,
            context=key_values,
            mask=key_value_mask,
            values_scale=key_value_scores,
            output_scale=query_scores,
        )

        local_out = None
        if exists(self.local_attn):
            local_out = self.local_attn(x, mask=mask)

        need_route_queries = exists(query_indices)

        if not need_route_queries:
            out = attn_out

            if exists(local_out):
                local_out = rearrange(local_out, "b n d -> b 1 n d")
                out = torch.cat((local_out, out), dim=1)

            out = reduce(attn_out, "b e n d -> b n d", "mean")

            if exists(mask):
                out = out.masked_fill(~mask[..., None], 0.0)

        out = torch.zeros_like(x)
        counts = torch.zeros(x.shape[:-1], device=x.device)

        query_indices = rearrange(query_indices, "b g n -> b (g n)")
        attn_out = rearrange(attn_out, "b g n d -> b (g n) d")

        expanded_query_indices = repeat(
            query_indices, "b n -> b n d", d=x.shape[-1]
        )
        attn_out_summed = out.scatter_add(1, expanded_query_indices, attn_out)

        ones = torch.ones(attn_out.shape[:-1], device=self.device)

        if exists(query_mask):
            ones = ones * rearrange(query_mask, "b g n -> b (g n)")

        counts = counts.scatter_add(1, query_indices, ones)
        counts = rearrange(counts, "... -> ... 1")

        has_unrouted = not exists(local_out)

        if not has_unrouted:
            counts = counts + 1
            attn_out_summed = attn_out_summed + local_out
        else:
            not_routed_mask = counts == 0
            attn_out_summed = attn_out_summed.masked_fill(not_routed_mask, 0.0)

        out = attn_out_summed

        # average if needed
        if self.average_routed:
            out = out / counts.clamp(min=1e-5)

        # for the position that were not routed, use a learned routing token
        # instead of just 0s
        if has_unrouted:
            out = torch.where(not_routed_mask, self.null_routed_token, out)

        if exists(mask):
            out = out.masked_fill(~mask[..., None], 0.0)

        return out


class MixtureOfAutoregressiveAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_routed_queries,
        num_routed_key_values,
        local_attn_window_size,
        routed_window_size=None,
        num_experts=2,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_triton=False,
        flash_attn=True,
        prenorm=True,
        average_routed=False,
        **kwargs,
    ):
        super().__init__()
        self.num_routed_queries = num_routed_queries
        self.num_routed_key_values = num_routed_key_values

        self.num_experts = num_experts
        self.null_tokens = nn.Parameter(torch.randn(num_experts, dim))
        routed_window_size = default(routed_window_size, local_attn_window_size)

        self.routed_window_size = routed_window_size
        self.average_routed = average_routed

        self.local_attn = LocalMHA(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            prenorm=prenorm,
            causal=True,
            window_size=local_attn_window_size,
        )

        self.query_router = CoordinateDescentRouter(
            dim, num_routing_tokens=num_experts, use_triton=use_triton, **kwargs
        )

        self.key_value_tensor = CoordinateDescentRouter(
            dim, num_routing_tokens=num_experts, use_triton=use_triton, **kwargs
        )

        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            groups=num_experts,
            dropout=dropout,
            flash=flash_attn,
            prenorm=prenorm,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        rotary_emb=None,
        num_routed_queries=None,
        num_routed_key_values=None,
    ):
        b = x.shape[0]
        w = self.routed_window_size
        num_windows = math.ceil(x.shape[-2] / w) - 1

        # cal local attn
        local_out = self.local_attn(x)

        if num_windows == 0:
            return local_out

        # pad sequence to multiple of routing window size
        mask = torch.ones(x.shape[:-1], device=self.device, dtype=torch.bool)
        x, seq_len = pad_to_multiple(x, w, dim=-2)
        mask, _ = pad_to_multiple(mask, w, dim=-1, value=False)

        context = x[..., :-w, :]
        context = repeat(context, "b n d -> (b nw) n d", mw=num_windows)

        context_mask = torch.ones(
            (num_windows, num_windows), device=self.device, dtype=torch.bool
        ).tril()
        context_mask = repeat(context_mask, "n1 n2 -> (b  n1) (n2 w)", b=b, w=w)

        # fold queries and mask into windows
        x = rearrange(x, "b (n w) d -> b n w d", w=w)
        mask = rearrange(mask, "b (n w) -> b n w", w=w)

        # omit the first window of queries as they have nothing to attend to
        x = rearrange(x[:, 1:, ...], "b n w d -> (b n) w d")
        mask = rearrange(mask[:, 1:, ...], "b n w -> (b n) w")

        # gets number of queries and key values to route
        num_routed_queries = default(
            num_routed_queries, self.num_routed_queries
        )
        num_routed_key_values = default(
            num_routed_key_values, self.num_routed_key_values
        )

        # coordinate descent routing
        query_indices, query_scores, queries, query_mask = self.query_router(
            x, mask=mask, num_tokens=num_routed_queries, keep_one_route_dim=True
        )
        query_scores = rearrange(query_scores, "b g n -> b g n 1")

        (
            kv_indices,
            key_value_scores,
            key_values,
            key_value_mask,
        ) = self.key_value_router(
            context,
            mask=context_mask,
            num_tokens=num_routed_key_values,
            keep_one_route_dim=True,
        )
        key_value_scores = rearrange(key_value_scores, "b g n -> b g 1 n 1")

        if exists(rotary_emb):
            rotary_emb, _ = pad_to_multiple(rotary_emb, w, dim=-2)

            windowed_rotary_emb = rearrange(rotary_emb, "(n w) d -> n w d", w=w)
            windowed_rotary_emb = windowed_rotary_emb[1:]
            windowed_rotary_emb = repeat(
                windowed_rotary_emb,
                "n w d -> (b n) g w d",
                b=b,
                g=query_scores.shape[1],
            )

            if exists(query_indices):
                rotary_query_indices = repeat(
                    query_indices,
                    "... -> ... d",
                    d=windowed_rotary_emb.shape[-1],
                )
                q_rotary_emb = windowed_rotary_emb.gather(
                    2, rotary_query_indices
                )

            else:
                q_rotary_emb = windowed_rotary_emb

            k_rotary_emb = (
                rotary_emb[kv_indices]
                if exists(kv_indices)
                else rotary_emb[: context.shape[-2]]
            )
            rotary_emb = (q_rotary_emb, k_rotary_emb)

        attn_out = self.attn(
            queries,
            rotary_emb=rotary_emb,
            context=key_values,
            mask=key_value_mask,
            values_scale=key_value_scores,
            output_scale=query_scores,
        )

        need_route_queries = exists(query_indices)

        if not need_route_queries:
            out = F.pad(attn_out, (0, 0, w, 0), value=0.0)
            out = out[:, :, :seq_len]

            if exists(local_out):
                local_out = rearrange(local_out, "b n d -> b 1 n d")
                out = torch.cat((local_out, out), dim=1)

            out = reduce(
                out,
                "b e n d -> b n d",
                "mean" if self.averaged_routed else "sum",
            )

        out = torch.zeros(
            (x.shape[0], self.num_experts, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )

        counts = torch.zeros(
            (x.shape[0], self.num_experts, x.shape[-2]), device=x.device
        )

        ones = torch.ones(attn_out.shape[:-1], device=self.device)

        if exists(query_mask):
            ones = ones * query_mask

        counts = counts.scatter_add(2, query_indices, ones)

        expanded_query_indices = repeat(
            query_indices, "b g n -> b g n d", d=x.shape[-1]
        )

        attn_out_summed = out.scatter_add(2, expanded_query_indices, attn_out)

        # for the positions that were not routed fill each with individual
        # expert null token
        fill_null_tokens = counts == 0 & ~rearrange(mask, "b n -> b 1 n")

        attn_out_summed = torch.where(
            rearrange(fill_null_tokens, "... -> ... 1"),
            rearrange(self.null_tokens, "g d -> 1 g 1 d"),
            attn_out_summed,
        )

        # un window the attention output as well as the routed counts
        attn_out_summed = rearrange(
            attn_out_summed, "(b n) g w d -> b g (n w) d", b=b
        )

        attn_out_summed = F.pad(attn_out_summed, (0, 0, w, 0), value=0.0)

        attn_out_summed = attn_out_summed[..., :seq_len, :]

        # local attended tokens with routed otkens
        attn_out_summed = reduce(attn_out_summed, "b g n d -> b n d", "sum")

        attn_out_summed = attn_out_summed + local_out

        # in expert seems to perform better while averaging
        if not self.averaged_routed:
            return attn_out_summed

        # avg tokens
        return attn_out_summed / (self.num_experts + 1)
