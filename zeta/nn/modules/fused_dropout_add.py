import torch


@torch.jit.script
def jit_dropout_add(x, residual, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return torch.nn.functional.dropout(x, p=prob, training=True) + residual


def fused_dropout_add(x, residual, prob, is_training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    if is_training:
        out = jit_dropout_add(x, residual, prob)
    else:
        out = (
            torch.nn.functional.dropout(x, p=prob, training=is_training)
            + residual
        )
    return out


@torch.jit.script
def jit_bias_dropout_add(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return (
        torch.nn.functional.dropout(x + bias, p=prob, training=True) + residual
    )


def fused_bias_dropout_add(x, bias, residual, prob, is_training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    if is_training:
        out = jit_bias_dropout_add(x, bias, residual, prob)
    else:
        out = (
            torch.nn.functional.dropout(x + bias, p=prob, training=is_training)
            + residual
        )
    return out
