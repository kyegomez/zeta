from einops import rearrange
from torch import Tensor, nn


class Film(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) module.

    This module applies feature-wise linear modulation to the input features based on the conditioning tensor.
    It scales and shifts the input features to adapt them to the given conditions.

    Args:
        dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layer in the network.
        expanse_ratio (int, optional): The expansion ratio for the hidden layer. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples::
        # Initialize the Film layer
        film_layer = Film(dim=128, hidden_dim=64, expanse_ratio=4)

        # Create some dummy data for conditions and hiddens
        conditions = torch.randn(10, 128)  # Batch size is 10, feature size is 128
        hiddens = torch.randn(10, 1, 128)  # Batch size is 10, sequence length is 1, feature size is 128

        # Pass the data through the Film layer
        modulated_features = film_layer(conditions, hiddens)

        # Print the shape of the output
        print(modulated_features.shape)  # Should be [10, 1, 128]
    """

    def __init__(
        self, dim: int, hidden_dim: int, expanse_ratio: int = 4, *args, **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.expanse_ratio = expanse_ratio

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * expanse_ratio),
            nn.SiLU(),
            nn.Linear(hidden_dim * expanse_ratio, dim * 2),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions: Tensor, hiddens: Tensor):
        """
        Forward pass of the FiLM module.

        Applies feature-wise linear modulation to the input features based on the conditioning tensor.

        INPUT SHAPE: [B, D]
        OUTPUT SHAPE: [B, 1, D]


        Args:
            conditions (Tensor): The conditioning tensor.
            hiddens (Tensor): The input features to be modulated.

        Returns:
            Tensor: The modulated features.
        """
        scale, shift = self.net(conditions).chunk(2, dim=-1)
        assert scale.shape[-1] == hiddens.shape[-1], (
            f"unexpected hidden dimension {hiddens.shape[-1]} used for"
            " conditioning"
        )
        scale, shift = map(
            lambda t: rearrange(t, "b d -> b 1 d"), (scale, shift)
        )
        return hiddens * (scale + 1) + shift


# # Initialize the Film layer
# film_layer = Film(dim=128, hidden_dim=64, expanse_ratio=4)

# # Create some dummy data for conditions and hiddens
# conditions = torch.randn(10, 128)  # Batch size is 10, feature size is 128
# hiddens = torch.randn(10, 1, 128)  # Batch size is 10, sequence length is 1, feature size is 128

# # Pass the data through the Film layer
# modulated_features = film_layer(conditions, hiddens)

# # Print the shape of the output
# print(modulated_features.shape)  # Should be [10, 1, 128]
