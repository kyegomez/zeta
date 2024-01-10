from torch import nn
from einops.layers.torch import EinMix as Mix


class VisionWeightedPermuteMLP(nn.Module):
    """
    VisionWeightedPermuteMLP module applies weighted permutation to the input tensor
    based on its spatial dimensions (height and width) and channel dimension.

    Args:
        H (int): Height of the input tensor.
        W (int): Width of the input tensor.
        C (int): Number of channels in the input tensor.
        seg_len (int): Length of each segment to divide the channels into.

    Attributes:
        mlp_c (Mix): MLP module for channel dimension permutation.
        mlp_h (Mix): MLP module for height dimension permutation.
        mlp_w (Mix): MLP module for width dimension permutation.
        proj (nn.Linear): Linear projection layer.

    """

    def __init__(self, H, W, C, seg_len):
        super().__init__()
        assert (
            C % seg_len == 0
        ), f"can't divide {C} into segments of length {seg_len}"
        self.mlp_c = Mix(
            "b h w c -> b h w c0",
            weight_shape="c c0",
            bias_shape="c0",
            c=C,
            c0=C,
        )
        self.mlp_h = Mix(
            "b h w (n c) -> b h0 w (n c0)",
            weight_shape="h c h0 c0",
            bias_shape="h0 c0",
            h=H,
            h0=H,
            c=seg_len,
            c0=seg_len,
        )
        self.mlp_w = Mix(
            "b h w (n c) -> b h w0 (n c0)",
            weight_shape="w c w0 c0",
            bias_shape="w0 c0",
            w=W,
            w0=W,
            c=seg_len,
            c0=seg_len,
        )
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        """
        Forward pass of the VisionWeightedPermuteMLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, C, H, W).

        """
        x = self.mlp_c(x) + self.mlp_h(x) + self.mlp_w(x)
        return self.proj(x)
