import torch
import torch.nn as nn

if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        print(
            "Triton is not installed. Please install it using `pip install"
            " triton`."
        )


@triton.jit
def linear_projection_kernel(
    X, W, Y, M, N, K, stride_x, stride_w, stride_y, BLOCK_SIZE: tl.constexpr
):
    # Compute indices
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Offsets for X, W, and Y
    x_off = row_idx * stride_x
    w_off = col_idx * stride_w
    y_off = row_idx * stride_y + col_idx

    # Dot product
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        acc += tl.load(X + x_off + k) * tl.load(W + w_off + k)
    tl.store(Y + y_off, acc)


class LinearTriton(nn.Module):
    """
    A custom linear module implemented using Triton.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to True, the module has a learnable bias. Default is True.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(LinearTriton, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Performs a forward pass through the linear module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Prepare the output tensor
        output = torch.empty(
            x.shape[0], self.out_features, device=x.device, dtype=x.dtype
        )

        # Grid and block dimensions
        grid = (x.shape[0], self.out_features)
        block = 128  # Example block size

        # Launch the Triton kernel
        linear_projection_kernel[grid](
            x,
            self.weight,
            output,
            x.shape[0],
            self.out_features,
            self.in_features,
            x.stride(0),
            self.weight.stride(0),
            output.stride(0),
            block,
        )

        # Add bias if present
        if self.bias is not None:
            output += self.bias.unsqueeze(0)  # Broadcasting the bias
        return output


# # Example usage
# model = LinearTriton(128, 64).cuda()
# input_tensor = torch.randn(1, 10, 128).cuda()
# output_tensor = model(input_tensor)
# print(output_tensor.shape)  # Should be torch.Size([10, 64])
