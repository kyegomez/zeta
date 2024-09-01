from torch import nn, Tensor


def exists(val):
    return val is not None


class AdaptiveGating(nn.Module):
    def __init__(self, hidden_dim: int):
        """
        Initializes an instance of the AdaptiveGating class.

        Args:
            hidden_dim (int): The dimension of the hidden state.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hat_text: Tensor,
        bar_text: Tensor,
    ) -> Tensor:
        """
        Performs the forward pass of the AdaptiveGating module.

        Args:
            hat_text (Tensor): The input tensor representing the hat text.
            bar_text (Tensor): The input tensor representing the bar text.

        Returns:
            Tensor: The fused hidden state tensor.

        """
        g = self.sigmoid(hat_text)

        # Step 2
        h_fused = bar_text * g + hat_text * (1 - g)

        return h_fused
