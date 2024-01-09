import torch
from torch import nn
from zeta.nn.modules.feedforward import FeedForward


class MixtralExpert(nn.Module):
    """

    At every layer, for every token, a router
    network chooses two of these groups (the “experts”) to process the token and combine their output
    additively. This technique increases the number of parameters of a model while controlling cost and
    latency, as the model only uses a fraction of the total set of parameters per token

    Args:
        dim (int):
        dim_out (int):
        num_experts (int):
        dropout (float, optional): Defaults to 0.0.


    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_experts: int,
        dropout: float = 0.0,
        expansion_rate: int = 2,
        *args,
        **kwargs,
    ):
        super(MixtralExpert, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_experts = num_experts
        self.dropout = dropout
        self.expansion_rate = expansion_rate

        for _ in range(self.num_experts):
            self.experts = nn.ModuleList(
                [
                    FeedForward(dim, dim, expansion_rate, *args, **kwargs)
                    for _ in range(self.num_experts)
                ]
            )

    def forward(self, x):
        # 2 of the experts are chosen to process the token
        two_experts = torch.randperm(self.num_experts)[:2]

        # Initialize a list to store the outputs of the selected experts
        expert_outputs = []

        for expert_id in two_experts:
            # Apply the selected expert to the input
            expert_output = self.experts[expert_id](x)
            # Add the expert's output to the list
            expert_outputs.append(expert_output)

        # Stack the expert outputs along a new dimension
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # Compute the weighted average of the expert outputs
        x = expert_outputs.mean(dim=0)

        return x


# # 3d tensor for text
# x = torch.randn(1, 512, 768)

# model = MixtralExpert(768, 768, 6)
# print(model(x).shape)
