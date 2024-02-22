from typing import List

import torch.nn.functional as F
from torch import nn


class MLPProjectionFusion(nn.Module):
    def __init__(
        self,
        input_dims: List[int],
        dim: int,
        num_experts: int,
    ):
        """
        Initializes an instance of MLPProjectionFusion.

        Args:
            input_dims (List[int]): A list of input dimensions for each expert.
            dim (int): The dimension of the MLP layers.
            num_experts (int): The number of experts.

        """
        super().__init__()
        self.input_dims = input_dims
        self.dim = dim
        self.num_experts = num_experts

        # First layer MLP for each expert
        self.mlp_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for dim in input_dims]
        )

        # Shared second layer of mlp2
        self.mlp2 = nn.Linear(dim, dim)

    def forward(self, *expert_inputs):
        """
        Forward pass of the MLPProjectionFusion module.

        Args:
            *expert_inputs: Variable number of expert inputs.

        Returns:
            torch.Tensor: The fused output.

        Raises:
            AssertionError: If the number of inputs does not match the number of experts.

        """
        assert (
            len(expert_inputs) == self.num_experts
        ), "Number of inputs must match number of experts"

        # Process each expert input through its mlp1 and sum the results
        expert_projections = [
            self.mlp2(F.relu(self.mlp_layers[i](input)))
            for i, input in enumerate(expert_inputs)
        ]

        # Fused output
        fused_output = sum(expert_projections)

        return fused_output
