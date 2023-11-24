import math
from functools import partial
from itertools import zip_longest
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from vector_quantize_pytorch import RandomProjectionQuantizer

from zeta.structs.attn_layers import rotate_half
from zeta.nn.attention.attend import Attend
from zeta.nn.attention.local_attention_mha import LocalMHA
from zeta.nn.embeddings.rope import RotaryEmbedding

# constants
mlist = nn.ModuleList

Linear = partial(nn.Linear, bias=False)

LocalMHA = partial(LocalMHA, causal=True, prenorm=True)

# helper functions


def exists(val):
    return val is not None


def is_power_of_two(n):
    return math.log2(n).is_integer()


def all_unique(arr):
    return len(set(arr)) == len(arr)


def apply_fns(fns, tensors):
    return [fn(tensor) for fn, tensor in zip(fns, tensors)]


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs


# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1


def apply_rotary_pos_emb(pos, t, scale=1.0):
    seq_len = t.shape[-2]

    pos = pos[..., -seq_len:, :]
    if not isinstance(scale, (int, float)):
        scale = scale[..., -seq_len:, :]

    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


def apply_rotary_pos_emb_qk(rotary_emb, q, k):
    freqs, scale = rotary_emb
    q = apply_rotary_pos_emb(freqs, q, scale)
    k = apply_rotary_pos_emb(freqs, k, scale**-1)
    return q, k


# token shift, from Peng et al of RWKV


def token_shift(t):
    t, t_shift = t.chunk(2, dim=-1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim=-1)


# hierarchy related classes


def pad_seq_to_multiple(t, mult):
    seq_len = t.shape[-2]
    next_seq_len_mult = math.ceil(seq_len / mult) * mult
    remainder = next_seq_len_mult - seq_len

    if remainder == 0:
        return t, seq_len

    t = F.pad(t, (0, 0, 0, remainder), value=0.0)
    return t, seq_len


def curtail_seq_to_multiple(t, mult):
    seq_len = t.shape[-2]
    prev_seq_len_mult = (seq_len // mult) * mult
    remainder = seq_len - prev_seq_len_mult

    if remainder == 0:
        return t

    t = t[..., :prev_seq_len_mult, :]
    return t


def hierarchical_cat(tokens, strides: Tuple[int, ...]):
    assert len(tokens) == len(strides)

    if all([s == 1 for s in strides]):
        return torch.cat(tokens, dim=-1)

    tokens = [
        repeat(t, "b n d -> b (n s) d", s=s) for t, s in zip(tokens, strides)
    ]
    min_seq_len = min([t.shape[-2] for t in tokens])
    tokens = [t[..., :min_seq_len, :] for t in tokens]
    return torch.cat(tokens, dim=-1)


class CausalConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1):
        super().__init__()
        self.causal_padding = kernel_size - 1
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, stride=stride)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)


class Compress(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        num_tokens=None,
        stride=1,
        compress_factor=1,
        expansion_factor=4,
        dim_head=64,
        heads=8,
        ignore_index=0,
        should_recon=False,
        should_prophet=False,
        prophet_num_predictions=None,
    ):
        super().__init__()
        assert compress_factor > 0 and is_power_of_two(compress_factor)

        self.stride = stride
        self.no_compress = compress_factor == 1
        self.compress_factor = compress_factor

        self.should_recon = should_recon
        self.should_prophet = should_prophet

        if self.no_compress:
            self.compress_fn = (
                Linear(dim, dim_out) if dim != dim_out else nn.Identity()
            )
            return

        dim_inner = int(dim * expansion_factor)

        self.compress_fn = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv(dim, dim_inner, compress_factor, stride=stride),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim_out, 1),
            Rearrange("b d n -> b n d"),
        )

        if should_recon:
            assert exists(num_tokens)
            self.to_recon = Linear(dim_out, compress_factor * num_tokens)

        if should_prophet:
            assert exists(prophet_num_predictions)
            self.to_prophet = Linear(dim_out, prophet_num_predictions)

        self.ignore_index = ignore_index

    def prophet(self, h, ids):
        if not self.should_prophet:
            return torch.zeros((), device=h.device).requires_grad_()

        c = self.compress_factor
        seq_len = ids.shape[-1]

        prophet_logits = self.to_prophet(h)
        prophet_logits = rearrange(
            prophet_logits, "b n (c d) -> (b c) d n", c=c
        )

        prophet_ids = F.pad(ids, (-1, c), value=self.ignore_index)
        prophet_ids = tuple(prophet_ids[:, i : (seq_len + i)] for i in range(c))
        prophet_ids = torch.stack(prophet_ids, dim=1)
        prophet_ids = rearrange(prophet_ids, "b c n -> (b c) n")

        if self.stride > 1:
            prophet_ids = prophet_ids[..., :: self.stride]

        prophet_loss = F.cross_entropy(
            prophet_logits, prophet_ids, ignore_index=self.ignore_index
        )
        return prophet_loss

    def recon(self, h, ids):
        assert self.should_recon

        if self.no_compress:
            return torch.zeros((), device=h.device).requires_grad_()

        c = self.compress_factor
        seq_len = ids.shape[-1]

        recon_logits = self.to_recon(h)
        recon_logits = rearrange(recon_logits, "b n (c d) -> (b c) d n", c=c)

        recon_ids = F.pad(ids, (c - 1, 0), value=self.ignore_index)
        recon_ids = tuple(recon_ids[:, i : (seq_len + i)] for i in range(c))
        recon_ids = torch.stack(recon_ids, dim=1)
        recon_ids = rearrange(recon_ids, "b c n -> (b c) n")

        if self.stride > 1:
            recon_ids = recon_ids[..., :: self.stride]

        recon_loss = F.cross_entropy(
            recon_logits, recon_ids, ignore_index=self.ignore_index
        )
        return recon_loss

    def forward(self, x):
        return self.compress_fn(x)


class HierarchicalMerge(nn.Module):
    def __init__(self, dims: Tuple[int, ...], dim_out, h_strides=1):
        super().__init__()
        dim = sum(dims)

        strides = cast_tuple(h_strides, len(dims))
        assert len(strides) == len(dims)

        self.strides = strides

        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_out * 2),
            nn.SiLU(),
            nn.Linear(dim_out * 2, dim_out),
        )

    def forward(self, tokens):
        x = hierarchical_cat(tokens, self.strides)
        return self.net(x)


# classes


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        dim_inner = int(dim * mult)

        self.net = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_inner),
            nn.GELU(),
            Linear(dim_inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, use_flash_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal=True, use_flash_attn=use_flash_attn)

        self.to_qkv = Linear(dim, dim_inner * 3)
        self.to_out = Linear(dim_inner, dim)

    def forward(self, x):
        n = x.shape[-2]
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        rotary_emb = self.rotary_emb(n)
        q, k = apply_rotary_pos_emb_qk(rotary_emb, q, k)

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class HierarchicalBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        window_size=None,
        compress_factor=1,
        stride=1,
        ff_mult=4,
    ):
        super().__init__()
        self.stride = stride

        assert is_power_of_two(compress_factor)
        self.compress_factor = compress_factor
        self.no_compress = compress_factor == 1

        assert not exists(window_size) or window_size >= 0
        self.has_attn = window_size != 0

        self.attn = None

        if self.has_attn:
            attn_klass = Attention
            if exists(window_size):
                attn_klass = partial(LocalMHA, window_size=window_size)

            self.attn = attn_klass(dim=dim, dim_head=dim_head, heads=heads)

        self.ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, x):
        c = self.compress_factor
        axial_dim = c // self.stride

        x, orig_seq_len = pad_seq_to_multiple(x, axial_dim)

        # hierarchical attention is performed with a simple axial attention

        # this, and using a convolution for compressing at the beginning
        # is one of the improvements on top of hourglass transformer
        # the downside is that the savings are only O(c) instead of O(c ** 2) as in hourglass transformer
        # you can get the O(c ** 2) saving by setting the hierarchical stride
        # == c, but you'll see that performance is much worse, as some tokens
        # will have a c - 1 token gap to the last hierarchical token

        if not self.no_compress:
            x = rearrange(x, "b (n c) d -> (b c) n d", c=axial_dim)

        if exists(self.attn):
            x = self.attn(token_shift(x)) + x

        x = self.ff(token_shift(x)) + x

        if not self.no_compress:
            x = rearrange(x, "(b c) n d -> b (n c) d", c=axial_dim)

        return x[:, :orig_seq_len]


class HierarchicalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        seq_len=2048,
        dim_head=64,
        heads=8,
        ff_mult=4,
        hierarchies=1,
        window_sizes=None,
        hierarchical_stride=1,
        hierarchy_merge_all=False,
        # whether to pass the pooled hierarchical information back to all
        # hierarchies or just one doing the prediction
        ignore_index=0,
        use_flash_attn=False,
        recon_loss_weight=0.1,
        prophet_loss_weight=0.0,
        prophet_loss_use_quantized=False,
        # for prophet, whether to use the next 1x token ids, or use the ids
        # from random projection quantization
        prophet_quantized_use_embed=False,
        predict_hierarchy=None,
        predict_use_all_hierarchy=False,
        rq_num_codebooks=4,
        rq_codebook_dim=256,
        rq_codebook_size=1024,
    ):
        """
        By not specifying hierarchies and window_sizes, you basically default to a regular autoregressive transformer with attention across full sequence length,
        Three hierarchies, all servicing predicting the next token

        from zeta.nn import HierarchicalTransformer

        model = HierarchicalTransformer(
            num_tokens = 256,
            dim = (128, 256, 512, 1024),
            depth = 8,
            seq_len = 1024,
            use_flash_attn = True,
            ff_mult = (2, 2, 4, 4),
            dim_head = (16, 32, 64, 64),
            heads = (2, 4, 8, 8),
            hierarchies = (1, 2, 4, 16),
            hierarchical_stride = (1, 1, 1, 8),  # this would determine the stride when compressing, and when concatting the hierarchical tokens to the fine tokens, the past tokens will be repeated this amount of time. causality is not violated as using the trick from hourglass transformers where sequence is shifted by compression factor - 1. recommend sticking with 1 except for highly compressed hierarchies, as it becomes very uncompetitive with baseline and generations look off
            window_sizes = (16, 32, 64, None)
        ).cuda()

        # hierarchies
        # 1x - dim 128 - attention (2 heads, 16 dim, receptive field 16)
        # 2x - dim 256 - attention (4 heads, 32 dim, receptive field 32)
        # 4x - dim 512 - attention (8 heads, 64 dim, receptive field 64)
        # 8x - dim 1024 - attention (8 heads, 64 dim, receptive field of all)
        """
        super().__init__()
        self.seq_len = seq_len

        hierarchies = cast_tuple(hierarchies)
        assert all_unique(
            hierarchies
        ), "hierarchies compression factors must be all unique integers"
        assert all(
            [*map(is_power_of_two, hierarchies)]
        ), "only powers of two allowed for hierarchies"

        self.hierarchies = hierarchies

        # just use a simple tuple list per hyperparameter to customize each
        # hierarchy

        num_hierarchies = len(hierarchies)

        dims = cast_tuple(dim, num_hierarchies)
        assert len(dims) == num_hierarchies

        window_sizes = cast_tuple(window_sizes, num_hierarchies)
        assert len(window_sizes) == num_hierarchies

        dim_head = cast_tuple(dim_head, num_hierarchies)
        assert len(dim_head) == num_hierarchies

        heads = cast_tuple(heads, num_hierarchies)
        assert len(heads) == num_hierarchies

        ff_mult = cast_tuple(ff_mult, num_hierarchies)
        assert len(ff_mult) == num_hierarchies

        hierarchical_stride = cast_tuple(hierarchical_stride, num_hierarchies)

        assert all(
            [*map(is_power_of_two, hierarchical_stride)]
        ), "all hierarchical strides must be power of two"
        assert all(
            [s <= h for s, h in zip(hierarchical_stride, hierarchies)]
        ), (
            "all strides must be less than the compression factor of the"
            " hierarchy"
        )

        self.h_strides = hierarchical_stride

        assert len(hierarchical_stride) == num_hierarchies

        # this determines to which hierarchy is everything pooled into for final prediction
        # however, final next token prediction can also use all hierarchies
        # with `predict_use_all_hierarchy`

        predict_hierarchy = default(predict_hierarchy, min(hierarchies))
        self.predict_hierarchy_index = hierarchies.index(predict_hierarchy)
        hierarchy_predict_dim = dims[self.predict_hierarchy_index]

        self.hierarchy_merge_all = hierarchy_merge_all
        assert (
            hierarchy_merge_all
            or self.h_strides[self.predict_hierarchy_index] == 1
        ), (
            "the hierarchy level being used for final next token prediction"
            " must have compression stride of 1"
        )

        # training related loss weights

        self.recon_loss_weight = recon_loss_weight
        self.prophet_loss_weight = prophet_loss_weight

        should_recon = recon_loss_weight > 0
        should_prophet = prophet_loss_weight > 0

        self.should_recon = should_recon
        self.should_prophet = should_prophet

        self.prophet_loss_use_quantized = prophet_loss_use_quantized
        self.prophet_quantized_use_embed = prophet_quantized_use_embed

        # token embedding

        dim_token_emb = max(dims)
        self.token_emb = nn.Embedding(num_tokens, dim_token_emb)

        # hierarchy compressions - 1x just uses the base token_emb weights

        self.compressors = mlist([])

        for dim, hierarchy, stride in zip(
            dims, hierarchies, hierarchical_stride
        ):
            self.compressors.append(
                Compress(
                    dim=dim_token_emb,
                    dim_out=dim,
                    num_tokens=num_tokens,
                    compress_factor=hierarchy,
                    stride=stride,
                    should_recon=should_recon,
                    should_prophet=should_prophet,
                    prophet_num_predictions=(
                        (hierarchy * num_tokens)
                        if not prophet_loss_use_quantized
                        else (rq_num_codebooks * rq_codebook_size)
                    ),
                )
            )

        # post token embedding norms

        self.post_token_emb_norms = mlist([nn.LayerNorm(dim) for dim in dims])

        # layers

        self.layers = mlist([])

        self.dims = dims

        self.hierarchical_merges = mlist([])
        self.need_hierarchical_merge = num_hierarchies > 1

        for _ in range(depth):
            hierarchical_layer = mlist([])

            # add a transformer block for each layer in the hierarchy

            for (
                hierarchy,
                h_stride,
                h_dim,
                h_window_size,
                h_dim_head,
                h_heads,
                h_ff_mult,
            ) in zip(
                hierarchies,
                hierarchical_stride,
                dims,
                window_sizes,
                dim_head,
                heads,
                ff_mult,
            ):
                # make sure the window size never exceeds the effective
                # sequence length

                effective_seq_len = seq_len // hierarchy

                if exists(h_window_size) and h_window_size > effective_seq_len:
                    print(
                        f"window size for hierarchy {hierarchy}x is greater"
                        " than effective sequence length - setting window size"
                        " to None (which would use normal full attention)"
                    )
                    h_window_size = None

                # add attention and feedforward

                hierarchical_layer.append(
                    HierarchicalBlock(
                        dim=h_dim,
                        dim_head=h_dim_head,
                        heads=h_heads,
                        window_size=h_window_size,
                        compress_factor=hierarchy,
                        stride=h_stride,
                        ff_mult=h_ff_mult,
                    )
                )

            self.layers.append(hierarchical_layer)

            # for merging the information across hierarchies
            # for now, only one direction, from all hierarchies to the
            # hierarchy that is being used to make predictions on, set by
            # predict_hierarchy_index above

            if not self.need_hierarchical_merge:
                continue

            merge = HierarchicalMerge(
                dims=dims,
                dim_out=(
                    hierarchy_predict_dim
                    if not self.hierarchy_merge_all
                    else sum(dims)
                ),
                h_strides=hierarchical_stride,
            )

            self.hierarchical_merges.append(merge)

        # final post-transformer norms, for all hierarchies

        self.norms = mlist([nn.LayerNorm(dim) for dim in dims])

        # random projection quantizer, for another approach to hierarchical
        # predictive coding

        if self.prophet_loss_use_quantized:
            rpq_klass = partial(
                RandomProjectionQuantizer,
                num_codebooks=rq_num_codebooks,
                codebook_dim=rq_codebook_dim,
                codebook_size=rq_codebook_size,
            )

            self.rand_proj_quantizers = mlist(
                [rpq_klass(dim=dim) for dim in dims]
            )
            self.rq_num_codebooks = rq_num_codebooks

        # to logit, for hierarchy set at predict_hierarchy_index, or all
        # hierarchies

        self.predict_use_all_hierarchy = predict_use_all_hierarchy
        logit_dim_in = (
            sum(dims) if predict_use_all_hierarchy else hierarchy_predict_dim
        )

        self.to_logits = Linear(logit_dim_in, num_tokens)

        # training related loss parameters

        self.ignore_index = ignore_index

    @torch.no_grad()
    @eval_decorator
    def generate(
        self, prompt, seq_len, temperature=1.0, filter_thres=0.9, **kwargs
    ):
        b, t, device = *prompt.shape, prompt.device

        out = prompt

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.seq_len :], **kwargs)[:, -1]
            filtered_logits = top_k(logits, thres=filter_thres)
            sample = gumbel_sample(filtered_logits, temperature=temperature)
            sample = rearrange(sample, "b -> b 1")
            out = torch.cat((out, sample), dim=-1)

        return out[:, t:]

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        ids,
        return_loss=False,
        return_hierarchical_token_embeds=False,
        return_hierarchical_embeds=False,
        ablate_hierarchical_merge=False,
        return_random_proj_quantize_ids=False,
    ):
        """
        einops notation:

        b - batch
        n - sequence length
        c - compression factor
        d - dimension
        """

        # if training, predict next token in sequence

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        # assert seq len

        assert ids.shape[-1] <= self.seq_len

        # get token embeddings, and pad to multiple of compression factor

        x = self.token_emb(ids)

        # for every hierarchy, compress token embeddings appropriately to the
        # hierarchical embeddings

        tokens = []

        for compress in self.compressors:
            tokens.append(compress(x))

        # save hierarchical tokens right before norm for random projection
        # quantization, if needed

        post_compressed_tokens = tokens

        # post embedding norms

        tokens = apply_fns(self.post_token_emb_norms, tokens)

        # if one wants all the compressed token embeds
        # just to investigate the space

        if return_hierarchical_token_embeds:
            return tokens

        # layers

        for layer, merge in zip_longest(self.layers, self.hierarchical_merges):
            tokens = apply_fns(layer, tokens)

            # pool the information all hierarchies
            # and then update the tokens that will be used to make the final
            # autoregressive prediction

            if not self.need_hierarchical_merge or ablate_hierarchical_merge:
                continue

            pooled = merge(tokens)

            if self.hierarchy_merge_all:
                tokens = [
                    (t + p[..., ::s, :])
                    for t, p, s in zip(
                        tokens, pooled.split(self.dims, dim=-1), self.h_strides
                    )
                ]
            else:
                predict_tokens = tokens[self.predict_hierarchy_index]
                predict_tokens = predict_tokens + pooled
                tokens[self.predict_hierarchy_index] = predict_tokens

        # final normalized embeddings

        embeds = apply_fns(self.norms, tokens)

        # if the researcher wants the randomly projected ids of either
        # compressed tokens or embeddings of the hierarchies

        if return_random_proj_quantize_ids:
            assert self.prophet_loss_use_quantized

            quantize_input = (
                embeds
                if self.prophet_quantized_use_embed
                else post_compressed_tokens
            )
            hierarchical_ids = apply_fns(
                self.rand_proj_quantizers, quantize_input
            )
            return hierarchical_ids

        # if one wants all the normalized hierarchical embeds

        if return_hierarchical_embeds:
            return embeds

        # select the hierarchical embeddings that will be doing the predicting

        if self.predict_use_all_hierarchy:
            predict_embed = hierarchical_cat(embeds, self.h_strides)
        else:
            predict_embed = embeds[self.predict_hierarchy_index]

        # logits for predicting next token

        logits = self.to_logits(predict_embed)

        if not return_loss:
            return logits

        ce_loss_fn = partial(F.cross_entropy, ignore_index=self.ignore_index)

        # autoregressive loss (predictive coding)

        logits = rearrange(logits, "b n c -> b c n")
        ce_loss = ce_loss_fn(logits, labels)

        # reconstruction losses for hierarchy tokens

        recon_losses = prophet_losses = torch.zeros(
            (), device=self.device
        ).requires_grad_()

        if self.should_recon:
            for compress, t in zip(self.compressors, embeds):
                recon_loss = compress.recon(t, ids)
                recon_losses = recon_losses + recon_loss

        # prophet losses for hierarchy tokens

        if self.should_prophet:
            if self.prophet_loss_use_quantized:
                # using random projected quantizer of the next hierarchical
                # token

                quantize_input = (
                    embeds
                    if self.prophet_quantized_use_embed
                    else post_compressed_tokens
                )

                hierarchical_ids = apply_fns(
                    self.rand_proj_quantizers, quantize_input
                )

                for hierarchy, stride, compress, embed, pred_ids in zip(
                    self.hierarchies,
                    self.h_strides,
                    self.compressors,
                    embeds,
                    hierarchical_ids,
                ):
                    if hierarchy == 1:
                        continue

                    prophet_logits = compress.to_prophet(embed)

                    axial_dim = hierarchy // stride

                    prophet_logits = curtail_seq_to_multiple(
                        prophet_logits, axial_dim
                    )
                    pred_ids = curtail_seq_to_multiple(pred_ids, axial_dim)

                    prophet_logits, pred_ids = map(
                        lambda t: rearrange(
                            t, "b (n c) ... -> (b c) n ...", c=axial_dim
                        ),
                        (prophet_logits, pred_ids),
                    )

                    prophet_logits = rearrange(
                        prophet_logits[:, :-1],
                        "b n (q c) -> (b q) c n",
                        q=self.rq_num_codebooks,
                    )
                    pred_ids = rearrange(pred_ids[:, 1:], "b n q -> (b q) n")

                    prophet_loss = ce_loss_fn(prophet_logits, pred_ids)
                    prophet_losses = prophet_losses + prophet_loss

            else:
                # or predicting the next N 1x base token ids
                # like prophetnet paper

                for compress, t in zip(self.compressors, embeds):
                    prophet_loss = compress.prophet(t, ids)
                    prophet_losses = prophet_losses + prophet_loss

        # total loss

        total_loss = (
            ce_loss
            + recon_losses * self.recon_loss_weight
            + prophet_losses * self.prophet_loss_weight
        )

        return total_loss, (ce_loss, recon_losses, prophet_losses)
