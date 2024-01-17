import math

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import functional as F
from torch.nn import init


def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError('version must be either 1, 2, or 3')
    batch, n = x.shape
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_reshaped = rearrange(x, 'b (k p) -> b k p', k=k)
    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        assert k == q == p == l == s == r == int(math.sqrt(n))
        return torch.einsum('bkp,kqp,qlk->blq', x_reshaped, w1_bfly, w2_bfly).reshape(batch, n)
    elif version == 2:  # Implementation 2
        out1 = torch.einsum('kqp,bkp->bkq', w1_bfly, x_reshaped)
        out1 = rearrange(rearrange(out1, 'b k q -> b (k q)'), 'b (r l) -> b l r', l=l)
        return torch.einsum('lsr,blr->bsl', w2_bfly, out1).reshape(batch, s * l)
    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        out1 = F.linear(x, w1_dense)
        out1 = rearrange(out1, 'b (r l) -> b (l r)', l=l)
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        out2 = F.linear(out1, w2_dense)
        out2 = rearrange(out2, 'b (l s) -> b (s l)', l=l)
        return out2


class BlockdiagButterflyMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly

blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply



def blockdiag_weight_to_dense_weight(weight):
    """
    Argumments:
        weight: (nblocks, out / nblocks, in / blocks)
    Return:
        dense_weight: (out / in)
    """
    return torch.block_diag(*torch.unbind(weight, dim=0))


def blockdiag_multiply_reference(x, weight):
    """
    This implementation is slow but more likely to be correct.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert nblocks * p == n

    x_reshaped = rearrange(x, '... (nblocks p) -> ... nblocks p', nblocks=nblocks)
    return rearrange(torch.einsum('...kp, kqp -> ...kq', x_reshaped, weight),
                     '... nblocks q -> ... (nblocks q)')


class BlockdiagMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
        out = torch.empty(batch_dim, nblocks, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out = torch.bmm(x_reshaped, weight.transpose(-1, -2), out=out).transpose(0, 1)
        return out.reshape(*batch_shape, nblocks * q)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        dx, dweight = None, None
        dout_reshaped = dout.reshape(batch_dim, nblocks, q).transpose(0, 1)
        if ctx.needs_input_grad[0]:
            dx = torch.empty(batch_dim, nblocks, p, device=x.device, dtype=x.dtype)
            dx = torch.bmm(dout_reshaped, weight.conj(),
                           out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
            dweight = torch.bmm(dout_reshaped.transpose(-1, -2), x_reshaped.conj())
        return dx, dweight


blockdiag_multiply = BlockdiagMultiply.apply


# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py


def fftconv_ref(u_variable, k, D_variable, dropout_mask, gelu=True, k_rev=None, flashfft=None):
    # u.shape:   B H L
    seqlen = u_variable.shape[-1]

    if flashfft is not None:
        y = flashfft(u_variable.to(dtype=torch.bfloat16).contiguous(), k)
    else:
        fft_size = 2 * seqlen
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()
        u_f = torch.fft.rfft(u_variable.to(dtype=k.dtype), n=fft_size)

        if len(u_variable.shape) > 3:
            k_f = k_f.unsqueeze(1)

        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u_variable * D_variable

    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u_variable.dtype)
    else:
        return out.to(dtype=u_variable.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class Sin(nn.Module):
    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()

        init_tensor = torch.ones(1, dim)
        self.freq = (
            nn.Parameter(w * init_tensor)
            if train_freq
            else w * torch.ones(1, dim)
        )
        self.w_mod = w_mod

    def forward(self, x):
        return torch.sin(self.w_mod * self.freq * x)


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output
    
    
