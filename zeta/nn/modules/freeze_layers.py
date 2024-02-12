from torch import Module


def set_module_requires_grad(
    module: Module,
    requires_grad: bool,
):
    """
    Set the `requires_grad` attribute of all parameters in the given module.

    Args:
        module (Module): The module whose parameters' `requires_grad` attribute needs to be set.
        requires_grad (bool): The value to set for the `requires_grad` attribute.

    Returns:
        None
    """
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers(module):
    """
    Freezes all layers in the given module by setting their requires_grad attribute to False.

    Args:
        module (nn.Module): The module whose layers need to be frozen.
    """
    set_module_requires_grad(module, False)
