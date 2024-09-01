from torch import Tensor, nn

from zeta.nn.modules.feedforward import FeedForward
from zeta.nn.modules.moe_router import MoERouter


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts in the mixture.
        hidden_layers (int, optional): Number of hidden layers in the experts. Defaults to None.
        mechanism (str, optional): Routing mechanism for selecting experts. Defaults to "softmax".
        custom_feedforward (callable, optional): Custom feedforward function for the experts. Defaults to None.
        ff_mult (int, optional): Multiplier for the hidden layer dimension in the experts. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples:
        x = torch.randn(2, 4, 6)
        model = MixtureOfExperts(dim=6, num_experts=2, hidden_layers=[32, 64])
        output = model(x)
        print(output.shape)

    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_layers: int = None,
        mechanism: str = "softmax",
        custom_feedforward: callable = None,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.hidden_layers = hidden_layers
        self.mechanism = mechanism
        self.custom_feedforward = custom_feedforward

        self.router = MoERouter(
            self.dim,
            self.num_experts,
            self.hidden_layers,
            self.mechanism,
        )

        self.experts = nn.ModuleList()

        for _ in range(self.num_experts):
            if self.custom_feedforward:
                self.experts.append(
                    self.custom_feedforward(
                        dim=self.num_experts,
                        dim_out=self.dim,
                        mult=ff_mult,
                        *args,
                        **kwargs,
                    )
                )
            else:
                self.experts.append(
                    FeedForward(
                        dim=self.num_experts,
                        dim_out=self.dim,
                        mult=ff_mult,
                        *args,
                        **kwargs,
                    )
                )

    def forward(self, x: Tensor):
        """Forward pass.

        Input Shape: (B, SEQ_LEN, DIM) where SEQ_LEN is the sequence length and num experts is the input dimension.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Router
        router = self.router(x)

        # Then we send the router output to the experts
        for i in range(self.num_experts):
            expert = self.experts[i]
            x = expert(router)

        return x
