import torch
from torch import nn, Tensor


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization module.


    Args:
        num_features (int): number of features in the input tensor
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5

    Shape:
        - Input: (batch_size, num_features, seq_len)
        - Output: (batch_size, num_features, seq_len)

    Examples:
        >>> x = torch.randn(20, 5, 10)
        >>> layer_norm = AdaptiveLayerNorm(5)
        >>> y = layer_norm(x)
        >>> y.shape
        torch.Size([20, 5, 10])

    """

    def __init__(self, num_features, eps=1e-5, *args, **kwargs):
        super(AdaptiveLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError("num_features must be a positive integer value")
        if not isinstance(eps, float) or eps <= 0:
            raise ValueError("eps must be a positive float value")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the AdaptiveLayerNorm module.

        Args:
            x (Tensor): torch tensor of shape (batch_size, num_features, seq_len)

        Returns:
            Tensor: the normalized input tensor
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
