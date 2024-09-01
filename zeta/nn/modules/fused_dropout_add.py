import torch
from torch import Tensor


@torch.jit.script
def jit_dropout_add(x: Tensor, residual: Tensor, prob: float) -> Tensor:
    return torch.nn.functional.dropout(x, p=prob, training=True) + residual


def fused_dropout_add(
    x: Tensor, residual: Tensor, prob: float, is_training: bool
) -> Tensor:
    """
    Applies fused dropout and addition operation to the input tensors.

    Args:
        x (Tensor): The input tensor.
        residual (Tensor): The residual tensor.
        prob (float): The probability of dropping out elements.
        is_training (bool): Whether the model is in training mode or not.

    Returns:
        Tensor: The output tensor after applying fused dropout and addition.
    """
    if is_training:
        out = jit_dropout_add(x, residual, prob)
    else:
        out = (
            torch.nn.functional.dropout(x, p=prob, training=is_training)
            + residual
        )
    return out


@torch.jit.script
def jit_bias_dropout_add(
    x: Tensor, bias: Tensor, residual: Tensor, prob: float
) -> Tensor:
    """
    Applies dropout to the sum of input `x` and `bias`, and then adds the `residual`.

    Args:
        x (Tensor): The input tensor.
        bias (Tensor): The bias tensor.
        residual (Tensor): The residual tensor.
        prob (float): The probability of an element to be zeroed.

    Returns:
        Tensor: The output tensor after applying dropout and adding the residual.
    """
    return (
        torch.nn.functional.dropout(x + bias, p=prob, training=True) + residual
    )


def fused_bias_dropout_add(
    x: Tensor, bias: Tensor, residual: Tensor, prob: float, is_training: bool
) -> Tensor:
    """
    Applies fused bias, dropout, and addition operation to the input tensor.

    Args:
        x (Tensor): The input tensor.
        bias (Tensor): The bias tensor.
        residual (Tensor): The residual tensor.
        prob (float): The probability of an element to be zeroed during dropout.
        is_training (bool): Whether the model is in training mode or not.

    Returns:
        Tensor: The output tensor after applying the fused bias, dropout, and addition operation.
    """
    if is_training:
        out = jit_bias_dropout_add(x, bias, residual, prob)
    else:
        out = (
            torch.nn.functional.dropout(x + bias, p=prob, training=is_training)
            + residual
        )
    return out
