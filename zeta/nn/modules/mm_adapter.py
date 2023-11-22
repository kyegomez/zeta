import torch 
from torch import nn


class SkipConnection(nn.Module):
    """
    A helper class for implementing skip connections.
    """
    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2


class MultiModalAdapterDenseNetwork(nn.Module):
    """
    Multi-modal adapter dense network that takes a tensor of shape (batch_size, dim) and returns a tensor of shape (batch_size, dim).

    Flow:
    x -> norm -> linear 1 -> silu -> concate -> linear 2 -> skip connection -> output

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension.
        depth (int): The depth of the network.
        activation (nn.Module): The activation function.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: The forward pass of the network.

    Example:
        >>> from zeta.nn import MultiModalAdapterDenseNetwork
        >>> mm_adapter = MultiModalAdapterDenseNetwork(
        ...     dim=512,
        ...     hidden_dim=1024,
        ...     depth=3,
        ... )
        >>> output = mm_adapter(x)
        >>> print(output.shape)
        torch.Size([1, 1024, 512])
    
    
    """
    def __init__(
        self,
        dim: int = None,
        hidden_dim: int = None,
        depth: int = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = dim
        self.depth = depth
        self.activation = activation

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.silu = nn.SiLU()

        # Define layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, dim),
                )
            )
        self.skip_connections = SkipConnection()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Normalize input tensor
    #     x = self.norm(x)

    #     # Linear projection 2 times
    #     lin1, lin2 = self.proj(x), self.proj(x)
        
    #     # Apply activation function silu to lin1
    #     x = self.activation(lin1)

    #     # Concate x and lin2
    #     concated = torch.cat([x, lin2])

    #     # Linear projection
    #     out = self.proj(concated)


    #     # Add skip connection for the depth
    #     return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        """
        for layer in self.layers:
            x = self.skip_connections(x, layer(x))
            return x
        
