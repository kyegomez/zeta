import torch
from einops import Rearrange
from torch import nn

from zeta.nn.modules.mbconv import MBConv, Residual
from zeta.nn.attention.attend import Attend

class MaxVit(nn.Module):
    def __init__(
            self,
            *,
            num_classes,
            dim,
            depth,
            dim_head: int = 32,
            dim_conv_stem = None,
            window_size: int = 7,
            mbconv_expansion_rate: int = 4,
            mbconv_shrinkage_rate = 0.25,
            dropout = 0.01,
            channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), "depth needs to be tuple of integers indicating number of transformer blocks at that stage"

        #conv stem
        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(
                channels,
                dim_conv_stem, 
                3,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                dim_conv_stem,
                dim_conv_stem,
                3,
                padding=1
            )
        )

        #vars
        num_stages = len(depth)
        dims = tuple(
            map(
                lambda i: (2 ** i) * dim, range(num_stages)
            )
        )
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        #shorthand for window size for efficient block - griid like attention
        w = window_size

        #iterate through stages
        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(
            dim_pairs, depth
        )):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim 

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),
                    Residual(
                        Attend(
                            dim=layer_dim, 
                            dim_head=dim_head, 
                            dropout=dropout
                        )
                    )
                )