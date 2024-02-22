from torch import nn

from zeta.utils import default


def nearest_upsample(dim: int, dim_out: int = None):
    """Nearest upsampling layer.

    Args:
        dim (int): _description_
        dim_out (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )
