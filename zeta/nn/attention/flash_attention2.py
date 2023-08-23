import math

import torch
from einops import rearrange
from torch import einsum, nn
from torch.autograd.function import Function
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel

# constants
EPSILON = 1e-10

# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# flash attention forwards and backwards
# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf

class FlashAttentionFunction(Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 1 in the v2 paper """

        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b n -> b 1 1 n')

        if not exists(mask):
            col_masks = (None,) * num_col_tiles
            mask = (col_masks,) * num_row_tiles 
        else:
            mask = ((mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split(q_bucket_size, dim = -2)
            mask = tuple(((row_mask,) * num_col_tiles) if row_mask.shape[-1] == 1 else row_mask.split(k_bucket_size, dim = -1) for row_mask in mask)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            mask,
            all_row_sums.split(q_bucket_size, dim = -2),
            all_row_maxes.split(q_bucket_size, dim = -2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                row_mask
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(col_mask):
                    attn_weights.masked_fill_(~col_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_weights = torch.exp(attn_weights - new_row_maxes)

                if exists(col_mask):
                    exp_weights.masked_fill_(~col_mask, 0.)

                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                oc.mul_(exp_row_max_diff).add_(exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

            oc.div_(row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
                row_mask
            )

            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = torch.exp(attn_weights - lsec)

                if exists(col_mask):
                    p.masked_fill_(~col_mask, 0.)

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim = -1, keepdims = True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None

# main class

# just flash attention in plain pytorch
# it will be way slower than implementing it in CUDA
# for tinkering and educational purposes


class FlashAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        q_bucket_size = 512,
        k_bucket_size = 1024,
        parallel = False,
        mixed_precision = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.parallel = parallel
        self.mixed_precision = mixed_precision

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # memory efficient attention related parameters
        # can be overriden on forward
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

        if self.parallel:
            self.model = DataParallel(self)
        if self.mixed_precision:
            self.scaler = GradScaler()

    def forward(
        self,
        x,
        context = None,
        mask = None,
        q_bucket_size = None,
        k_bucket_size = None,
    ):
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if self.parallel:
            # Split the input data into chunks and move each chunk to the correct GPU
            num_gpus = torch.cuda.device_count()
            x_chunks = x.split(x.size(0) // num_gpus)
            x_chunks = [chunk.to(f'cuda:{i}') for i, chunk in enumerate(x_chunks)]
            q = x_chunks

        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast():
                out = FlashAttentionFunction.apply(q, k, v, mask, self.causal, q_bucket_size, k_bucket_size)
        else:
            out = FlashAttentionFunction.apply(q, k, v, mask, self.causal, q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)