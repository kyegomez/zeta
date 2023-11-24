import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Optional
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, nn
from torch.nn import Module

# helper


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


# helper classes


def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# adaptive conv from Karras et al. Stylegan2
# for conditioning on latents


class AdaptiveConv3DMod(Module):
    """
    Adaptive convolutional layer, with support for spatial modulation

    Args:
        dim: input channels
        spatial_kernel: spatial kernel size
        time_kernel: temporal kernel size
        dim_out: output channels
        demod: demodulate weights
        eps: epsilon for numerical stability

    Returns:
        Tensor of shape (batch, channels, time, height, width)

    Examples:
    >>> x = torch.randn(1, 512, 4, 4, 4)
    >>> mod = torch.randn(1, 512)
    >>> layer = AdaptiveConv3DMod(512, 3, 3)
    >>> out = layer(x, mod)
    >>> out.shape


    """

    @beartype
    def __init__(
        self,
        dim,
        *,
        spatial_kernel,
        time_kernel,
        dim_out=None,
        demod=True,
        eps=1e-8,
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.eps = eps

        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel
        self.time_kernel = time_kernel

        self.padding = (
            *((spatial_kernel // 2,) * 4),
            *((time_kernel // 2,) * 2),
        )
        self.weights = nn.Parameter(
            torch.randn(
                (dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)
            )
        )

        self.demod = demod

        nn.init.kaiming_normal_(
            self.weights, a=0, mode="fan_in", nonlinearity="selu"
        )

    def forward(self, fmap, mod: Optional[Tensor] = None):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b = fmap.shape[0]

        # prepare weights for modulation

        weights = self.weights

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, "b i -> b 1 i 1 1 1")

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = (
                reduce(weights**2, "b o i k0 k1 k2 -> b o 1 1 1 1", "sum")
                .clamp(min=self.eps)
                .rsqrt()
            )
            weights = weights * inv_norm

        fmap = rearrange(fmap, "b c t h w -> 1 (b c) t h w")

        weights = rearrange(weights, "b o ... -> (b o) ...")

        fmap = F.pad(fmap, self.padding)
        fmap = F.conv3d(fmap, weights, groups=b)

        return rearrange(fmap, "1 (b o) ... -> b o ...", b=b)
