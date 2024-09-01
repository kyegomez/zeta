from torch import nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        dim_in (int): The dimension of the input tensor.
        dim_out (int): The dimension of the output tensor.
        expansion_factor (float, optional): The expansion factor for the hidden dimension. Default is 2.
        depth (int, optional): The number of hidden layers. Default is 2.
        norm (bool, optional): Whether to apply layer normalization to the hidden layers. Default is False.

    Attributes:
        net (nn.Sequential): The sequential module for the MLP.

    #Usage
    ```
    from zeta.nn import MLP

    mlp = MLP(
        dim_in=256,
        dim_out=10,
        expansion_factor=4.,
        depth=3,
        norm=True
    )

    x = torch.randn(32, 256)

    #apply the mlp
    output = mlp(x)

    #output tensor
    print(output)
    ```

    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        expansion_factor=2.0,
        depth=2,
        norm=False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)

        def norm_fn():
            return nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [
            nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.SiLU(), norm_fn())
        ]

        for _ in range(depth - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), norm_fn()
                )
            )
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.net(x.float())
