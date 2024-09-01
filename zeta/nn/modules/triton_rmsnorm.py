import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride,
    n_cols,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < n_cols)
    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < n_cols)


@torch.inference_mode()
def trmsnorm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    """
    Applies the Triton RMSNorm operation to the given hidden states.

    Args:
        hidden_states (Tensor): The input hidden states.
        weight (Tensor): The weight tensor.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-6.

    Returns:
        Tensor: The output tensor after applying the RMSNorm operation.
    """

    def _kernel_meta():
        device = hidden_states.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)
    kernel_meta = _kernel_meta()
    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_stride,
        feat_size,
        eps,
        feat_size,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )


# Example input tensor
# hidden_states = torch.randn(10, 20, 30)
# weight = torch.randn(30)

# # Apply RMSNorm operation
# output = trmsnorm(hidden_states, weight)
