import torch
from einops import pack, rearrange, unpack
from torch import Tensor, nn

from zeta.nn.attention.spatial_linear_attention import SpatialLinearAttention
from zeta.nn.modules.img_or_video_to_time import image_or_video_to_time


def divisible_by(num, den):
    return (num % den) == 0


def exists(val):
    return val is not None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def compact_values(d: dict):
    return {k: v for k, v in d.items() if exists(v)}


def is_odd(n):
    return not divisible_by(n, 2)


def init_bilinear_kernel_1d(conv: nn.Module):
    nn.init.zeros_(conv.weight)
    if exists(conv.bias):
        nn.init.zeros_(conv.bias)

    channels = conv.weight.shape[0]
    bilinear_kernel = Tensor([0.5, 1.0, 0.5])
    diag_mask = torch.eye(channels).bool()
    conv.weight.data[diag_mask] = bilinear_kernel


class TemporalDownsample(nn.Module):
    """
    Temporal downsample module that reduces the time dimension of the input tensor by a factor of 2.

    Args:
        dim (int): The number of input channels.
        time_dim (int, optional): The index of the time dimension in the input tensor. If None, the last dimension is assumed to be the time dimension.

    Attributes:
        dim (int): The number of input channels.
        time_dim (int): The index of the time dimension in the input tensor.
        conv (nn.Conv1d): 1D convolutional layer used for downsampling.
    """

    def __init__(self, dim: int, time_dim: int = None, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim

        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

        init_bilinear_kernel_1d(self.conv)

    def forward(
        self,
        x: Tensor,
    ):
        """
        Forward pass of the temporal downsample module.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, ..., time_dim, dim).

        Returns:
            torch.Tensor: The downsampled tensor with shape (batch_size, ..., time_dim // 2, dim).

        Raises:
            AssertionError: If the time dimension of the input tensor is not greater than 1.
        """
        assert x.shape[-1] > 1, "time dimension must be greater than 1"
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """
    Upsamples the temporal dimension of the input tensor using transposed convolution.

    Args:
        dim (int): The number of input channels.
        time_dim (int, optional): The index of the temporal dimension. If None, the last dimension is assumed to be the temporal dimension.
    """

    def __init__(self, dim: int, time_dim: int = None):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim

        self.conv = nn.ConvTranspose1d(
            dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        init_bilinear_kernel_1d(self.conv)

    @image_or_video_to_time
    def forward(self, x: Tensor):
        """
        Performs forward pass through the TemporalUpsample module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, ..., dim, time).

        Returns:
            torch.Tensor: The upsampled tensor of shape (batch_size, ..., dim, 2*time).
        """
        return self.conv(x)


class ConvolutionInflationBlock(nn.Module):
    """
    Convolution Inflation Block module.

    Args:
        dim (int): Number of input channels.
        conv2d_kernel_size (int): Kernel size for the spatial convolution.
        conv1d_kernel_size (int): Kernel size for the temporal convolution.
        groups (int): Number of groups to use for group normalization.
        time_dim (int): Number of time steps in the input tensor.

    Attributes:
        dim (int): Number of input channels.
        conv2d_kernel_size (int): Kernel size for the spatial convolution.
        conv1d_kernel_size (int): Kernel size for the temporal convolution.
        groups (int): Number of groups to use for group normalization.
        time_dim (int): Number of time steps in the input tensor.
        spatial_conv (nn.Sequential): Sequential module for spatial convolution.
        temporal_conv (nn.Sequential): Sequential module for temporal convolution.
        proj_out (nn.Conv1d): 1D convolution layer for projection.

    Methods:
        forward(x, batch_size=None): Forward pass of the ConvolutionInflationBlock module.

    """

    def __init__(
        self,
        dim: int,
        conv2d_kernel_size: int = 3,
        conv1d_kernel_size: int = 3,
        groups: int = 8,
        time_dim: int = None,
    ):
        super().__init__()
        assert is_odd(conv2d_kernel_size), "conv2d_kernel_size must be odd"
        assert is_odd(conv1d_kernel_size), "conv1d_kernel_size must be odd"

        self.dim = dim
        self.conv2d_kernel_size = conv2d_kernel_size
        self.conv1d_kernel_size = conv1d_kernel_size
        self.groups = groups
        self.time_dim = time_dim

        # Self spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                conv2d_kernel_size,
                padding=conv2d_kernel_size // 2,
            ),
            nn.GroupNorm(groups, num_channels=dim),
            nn.SiLU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                conv1d_kernel_size,
                padding=conv1d_kernel_size // 2,
            ),
            nn.GroupNorm(groups, num_channels=dim),
            nn.SiLU(),
        )

        self.proj_out = nn.Conv1d(dim, dim, 1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: Tensor,
        batch_size: int = None,
    ):
        """
        Forward pass of the ConvolutionInflationBlock module.

        Args:
            x (Tensor): Input tensor.
            batch_size (int, optional): Batch size of the input tensor.

        Returns:
            Tensor: Output tensor after applying the ConvolutionInflationBlock.

        """
        residual = x
        is_video = x.ndim == 5

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")

        x = self.spatial_conv(x)

        rearrange_kwargs = compact_values(dict(b=batch_size, t=self.time_dim))

        assert (
            len(rearrange_kwargs) > 0
        ), "batch_size and time_dim must be provided"
        x = rearrange(x, "(b t) c h w -> b h w c t", **rearrange_kwargs)

        x, ps = pack_one(x, "* c t")

        x = self.temporal_conv(x)
        x = self.proj_out(x)

        x = unpack_one(x, ps, "* c t")

        if is_video:
            x = rearrange(x, "b h w c t -> b c t h w")
        else:
            x = rearrange(x, "b h w c t -> (b t) c h w")

        return x + residual


class AttentionBasedInflationBlock(nn.Module):
    """
    Attention-based inflation block module.

    Args:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout rate. Defaults to 0.1.

    Attributes:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        attn (SpatialLinearAttention): The spatial linear ablttention module.
        proj (nn.Linear): The linear projection layer.
        norm (nn.LayerNorm): The layer normalization module.

    Example:
        >>> import torch
        >>> from lumiere.model import AttentionBasedInflationBlock
        >>> x = torch.randn(1, 4, 224, 224, 512)
        >>> model = AttentionBasedInflationBlock(dim=512, heads=4, dropout=0.1)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 4, 224, 224, 512])

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout

        # Spatial linear attention for videos of size:
        # batch_size, channels, frames, height, width.
        self.attn = SpatialLinearAttention(
            dim, heads, dim_head=dim // heads, *args, **kwargs
        )

        # Linear projection layer
        self.proj = nn.Linear(dim, dim)

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the AttentionBasedInflationBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        skip = x
        b, t, h, w, d = x.shape

        # Reshape to match the spatial linear attention module
        x = rearrange(x, "b t h w d -> b d t h w")

        # Apply spatial linear attention
        x = self.attn(x)

        # Reshape back to the original shape
        x = rearrange(x, "b d t h w -> b t h w d")

        # Linear projection
        x = nn.Linear(d, d)(x)

        return x + skip
