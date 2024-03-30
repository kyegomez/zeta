import time
import math
import torch
import triton
import triton.language as tl


class Functions:
    @staticmethod
    @triton.jit
    def tanh_activation_kernel(
        x_ptr,
        out_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        exp2x = tl.exp(2 * x)
        output = 1 - 2 / (exp2x + 1)
        tl.store(out_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def hard_tanh_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        shape_condition = tl.where(x < -1, -1, x)
        output = tl.where(x > 1, 1, shape_condition)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def relu_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.maximum(0, x)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def relu6_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.minimum(tl.maximum(x, 0), 6.0)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def leaky_relu_activation_kernel(
        x_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.maximum(x, alpha * x)
        tl.store(output_ptr + offsets, output, mask=mask)
