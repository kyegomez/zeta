import torch
from torch import nn


class MixtureOfSoftmaxes(nn.Module):
    """
    Implements Mixture of Softmaxes (MoS) as described by Yang et al., 2017.
    This increases the expressiveness of the softmax by combining multiple softmaxes.

    Args:
        num_mixtures (int): Number of softmax mixtures.
        input_size (int): Size of the input feature dimension.
        num_classes (int): Number of classes (output dimension).

    Shape:
        - Input: (N, input_size)
        - Output: (N, num_classes)

    Examples:
        >>> x = torch.randn(32, 128)
        >>> mos = MixtureOfSoftmaxes(5, 128, 10)
        >>> output = mos(x)
        >>> print(output.shape)
        torch.Size([32, 10])
    """

    def __init__(self, num_mixtures, input_size, num_classes):
        super(MixtureOfSoftmaxes, self).__init__()
        self.num_mixtures = num_mixtures
        self.input_size = input_size
        self.num_classes = num_classes

        # Linear transformations for the mixture coefficients and softmaxes
        self.mixture_weights = nn.Linear(input_size, num_mixtures)
        self.softmax_layers = nn.ModuleList(
            [nn.Linear(input_size, num_classes) for _ in range(num_mixtures)]
        )

    def forward(self, x):
        """
        Forward pass for Mixture of Softmaxes.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Combined output from the mixture of softmaxes.
        """
        mixture_weights = torch.softmax(self.mixture_weights(x), dim=1)
        softmax_outputs = [
            torch.softmax(layer(x), dim=1) for layer in self.softmax_layers
        ]

        # Combine softmax outputs weighted by the mixture coefficients
        output = torch.stack(
            softmax_outputs, dim=1
        ) * mixture_weights.unsqueeze(2)
        return output.sum(dim=1)
