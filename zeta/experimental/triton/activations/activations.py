import torch
import triton
import triton.language as tl

from typing import Callable
from activations.functions import Functions

BLOCK_SIZE = 1024


def apply_activation(
    x: torch.Tensor, activation_fn: Callable[..., torch.Tensor], *args, **kwargs
):
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA.")

    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    activation_args = [x, output] + list(args)

    if "n_elements" not in kwargs:
        kwargs["n_elements"] = n_elements

    activation_fn[grid](*activation_args, BLOCK_SIZE=1024, **kwargs)

    return output


def tanh_activation(x: torch.Tensor):
    return apply_activation(x, Functions.tanh_activation_kernel)


def hard_tanh_activation(x: torch.Tensor):
    return apply_activation(x, Functions.hard_tanh_activation_kernel)


def relu_activation(x: torch.Tensor):
    return apply_activation(x, Functions.relu_activation_kernel)


def relu6_activation(x: torch.Tensor):
    return apply_activation(x, Functions.relu6_activation_kernel)


def leaky_relu_activation(x: torch.Tensor, alpha: float = 0.2):
    return apply_activation(
        x, Functions.leaky_relu_activation_kernel, alpha=alpha
    )


def smooth_relu_activation(x: torch.Tensor, beta: float = 2.0):
    # Make input tensor contiguous if needed
    if not x.is_contiguous():
        x = x.contiguous()

    return apply_activation(
        x, Functions.smooth_relu_activation_kernel, beta=beta
    )


def softsign_activation(x: torch.Tensor):
    return apply_activation(x, Functions.softsign_activation_kernel)


def softplus_activation(x: torch.Tensor):
    return apply_activation(x, Functions.softplus_activation_kernel)


def sigmoid_activation(x: torch.Tensor):
    return apply_activation(x, Functions.sigmoid_activation_kernel)


def hard_sigmoid_activation(x: torch.Tensor):
    return apply_activation(x, Functions.hard_sigmoid_activation_kernel)


def silu_activation(x: torch.Tensor):
    return apply_activation(x, Functions.silu_activation_kernel)


def hard_silu_activation(x: torch.Tensor):
    return apply_activation(x, Functions.hard_silu_activation_kernel)


def softmax_activation(x: torch.Tensor):
    return apply_activation(x, Functions.softmax_activation_kernel)


def gelu_activation(x: torch.Tensor, approximate: bool = True):
    return apply_activation(x, Functions.gelu_activation_kernel, approximate)


def swiglu_activation(x: torch.Tensor):
    return apply_activation(x, Functions.swiglu_activation_kernel)
