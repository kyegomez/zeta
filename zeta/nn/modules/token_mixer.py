from torch import nn
from einops.layers.torch import EinMix as Mix


def TokenMixer(
    num_features: int, n_patches: int, expansion_factor: int, dropout: float
):
    """
    TokenMixer module that performs token mixing in a neural network.

    Args:
        num_features (int): Number of input features.
        n_patches (int): Number of patches.
        expansion_factor (int): Expansion factor for hidden dimension.
        dropout (float): Dropout probability.

    Returns:
        nn.Sequential: TokenMixer module.
    """
    n_hidden = n_patches * expansion_factor
    return nn.Sequential(
        nn.LayerNorm(num_features),
        Mix(
            "b hw c -> b hid c",
            weight_shape="hw hid",
            bias_shape="hid",
            hw=n_patches,
            hidden=n_hidden,
        ),
        nn.GELU(),
        nn.Dropout(dropout),
        Mix(
            "b hid c -> b hw c",
            weight_shape="hid hw",
            bias_shape="hw",
            hw=n_patches,
            hidden=n_hidden,
        ),
        nn.Dropout(dropout),
    )

