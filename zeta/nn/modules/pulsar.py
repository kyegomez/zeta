import torch 
from torch import nn
import torch.nn.functional as F


class LogGammaActivation(torch.autograd.Function):
    """
    PulSar Activation function that utilizes factorial calculus

    PulSar Activation function is defined as:
        f(x) = log(gamma(x + 1))
    where gamma is the gamma function

    The gradient of the PulSar Activation function is defined as:
        f'(x) = polygamma(0, x + 2)
    where polygamma is the polygamma function

    Methods:
        forward(ctx, input): Computes the forward pass
        backward(ctx, grad_output): Computes the backward pass
    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the PulSar Activation function
        
        """
        #compute forward pass
        gamma_value = torch.lgamma(input + 1)
        ctx.save_for_backward(input, gamma_value)
        return gamma_value
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the PulSar Activation function
        """
        #compute gradient for backward pass
        input, gamma_value = ctx.saved_tensors
        polygamma_val = torch.polygamma(0, input + 2)
        return polygamma_val * grad_output
    

class Pulsar(nn.Module):
    """
    Pulsar Activation function that utilizes factorial calculus

    Pulsar Activation function is defined as:
        f(x) = log(gamma(x + 1))
    where gamma is the gamma function


    Usage:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    pulsar = Pulsar()
    y = pulsar(x)
    print(y)
    y = y.backward(torch.ones_like(x))


    
    """
    def forward(self, x):
        """
        Forward pass of the PulSar Activation function
        """
        return LogGammaActivation.apply(x)
    
