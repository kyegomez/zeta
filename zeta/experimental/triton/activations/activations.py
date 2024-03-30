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


def tanh_activation(x: torch.Tensor, *args, **kwargs):
    return apply_activation(
        x, Functions.tanh_activation_kernel, *args, **kwargs
    )


def hard_tanh_activation(x: torch.Tensor, *args, **kwargs):
    return apply_activation(
        x, Functions.hard_tanh_activation_kernel, *args, **kwargs
    )


def relu_activation(x: torch.Tensor, *args, **kwargs):
    return apply_activation(
        x, Functions.relu_activation_kernel, *args, **kwargs
    )


def relu6_activation(x: torch.Tensor, *args, **kwargs):
    return apply_activation(
        x, Functions.relu6_activation_kernel, *args, **kwargs
    )


def leaky_relu_activation(x: torch.Tensor, alpha: float = 0.2, *args, **kwargs):
    return apply_activation(
        x, Functions.leaky_relu_activation_kernel, alpha=alpha, *args, **kwargs
    )
