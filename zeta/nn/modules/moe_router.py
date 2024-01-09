import torch
from torch import nn, Tensor
import torch.nn.functional as F
from zeta.ops.sparsemax import sparsemax


class MoERouter(nn.Module):
    """
    MoERouter is a module that routes input data to multiple experts based on a specified mechanism.

    Args:
        dim (int): The input dimension.
        num_experts (int): The number of experts to route the data to.
        hidden_layers (int, optional): The number of hidden layers in the routing network. Defaults to None.
        mechanism (str, optional): The routing mechanism to use. Must be one of "softmax" or "gumbel". Defaults to "softmax".

    Raises:
        ValueError: If the mechanism is not "softmax" or "gumbel".

    Input Shape:
        (B, SEQ_LEN, DIM) where SEQ_LEN is the sequence length and DIM is the input dimension.

    Output Shape:
        (B, SEQ_LEN, NUM_EXPERTS) where NUM_EXPERTS is the number of experts.

    Example:
        >>> x = torch.randn(2, 4, 6)
        >>> router = MoERouter(dim=6, num_experts=2, hidden_layers=[32, 64])
        >>> output = router(x)
        >>> output.shape
        torch.Size([2, 4, 2])
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_layers: int = None,
        mechanism: "str" = "softmax",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.hidden_layers = hidden_layers
        self.mechanism = mechanism

        if hidden_layers:
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(self.dim, self.hidden_layers[0]))

            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.ReLU())
                self.layers.append(
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1])
                )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_layers[-1], self.num_experts))
        else:
            # self.layers = nn.ModuleList([nn.Linear(self.dim, self.num_experts)])
            self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim)])

    def forward(self, x: Tensor, *args, **kwargs):
        """
        Forward pass of the MoERouter module.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the routing mechanism applied to the input data.

        """
        for layer in self.layers:
            x = layer(x)

        if self.mechanism == "softmax":
            return F.softmax(x, dim=1)

        elif self.mechanism == "gumbel":
            return F.gumbel_softmax(x, hard=True)

        elif self.mechanism == "topk":
            return torch.topk(x, k=self.num_experts, dim=1)[1]

        elif self.mechanism == "sample":
            return torch.multinomial(x, num_samples=2, replacement=False)

        elif self.mechanism == "weighted_average":
            return x.mean(dim=0)

        elif self.mechanism == "gate":
            return torch.sigmoid(x)

        elif self.mechanism == "top1":
            return torch.topk(x, 1, dim=1)[1]

        elif self.mechanism == "sparsemax":
            return sparsemax(x)

        else:
            raise ValueError("Mechanism must be either softmax or gumbel")
