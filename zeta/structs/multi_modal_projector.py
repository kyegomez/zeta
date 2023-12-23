import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def build_vision_projector(config, delay_load=False, **kwargs):
    """
    Builds a vision projector based on the given configuration.

    Args:
        config: The configuration object containing the projector type and other parameters.
        delay_load: Whether to delay the loading of the projector.
        **kwargs: Additional keyword arguments.

    Returns:
        A vision projector module based on the specified projector type.

    Raises:
        ValueError: If the specified projector type is unknown.


    Example:
    >>> config = {"mm_projector_type": "identity"}
    >>> projector = build_vision_projector(config)
    >>> print(projector)
    IdentityMap()

    """
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
