import torch
import torch.nn as nn


class STEFunc(torch.autograd.Function):
    """
    Straight Through Estimator

    This function is used to bypass the non differentiable operations


    Args:
        input (torch.Tensor): the input tensor

    Returns:
        torch.Tensor: the output tensor

    Usage:
    >>> x = torch.randn(2, 3, requires_grad=True)
    >>> y = STEFunc.apply(x)
    >>> y.backward(torch.ones_like(x))
    >>> x.grad


    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the STE function where we clamp the input between -1 and 1 and then apply the sign function

        Args:
            ctx (torch.autograd.Function): the context object
            input (torch.Tensor): the input tensor



        """
        return torch.sign(torch.clamp(input, min=-1.0, max=1.0))

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the STE function where we bypass the non differentiable operations

        """
        # Bypass the non differterable operations
        return grad_output


class STE(nn.Module):
    """
    STE Module

    This module is used to bypass the non differentiable operations

    Args:
        input (torch.Tensor): the input tensor

    Returns:
        torch.Tensor: the output tensor

    Usage:
    >>> ste = STE()
    >>> x = torch.randn(2, 3, requires_grad=True)
    >>> y = ste(x)
    >>> y.backward(torch.ones_like(x))
    >>> x.grad

    """

    def forward(self, input):
        return STEFunc.apply(input)
