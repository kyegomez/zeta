import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zeta.nn.modules.feedforward import FeedForward as Expert


class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        ff_dim: int,
        top_k: int,
        num_shared_experts: int,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.ff_dim = ff_dim
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.ff_mult = ff_mult

        # Initialize the correct number of experts
        self.experts = nn.ModuleList(
            [
                Expert(dim, dim // num_experts, ff_mult, *args, **kwargs)
                for _ in range(num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(dim, dim, ff_mult, *args, **kwargs)
                for _ in range(num_shared_experts)
            ]
        )
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # Flatten for gating

        # Apply gating mechanism and ensure indices are within the valid range
        gate_scores = F.softmax(self.gate(x_flat), dim=-1)
        # Limit the number of experts to self.num_experts
        gate_scores = gate_scores[:, : self.num_experts]
        topk_val, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)

        # Process shared experts
        shared_output = sum([expert(x) for expert in self.shared_experts])

        # Process routed experts
        final_output = shared_output
        for i in range(self.top_k):
            expert_outputs = torch.stack(
                [self.experts[idx](x) for idx in topk_idx[:, i]], dim=2
            )  # Stack along a new dimension
            expert_weights = (
                topk_val[:, i].unsqueeze(-1).unsqueeze(-1)
            )  # Reshape for broadcasting
            expert_output = torch.sum(
                expert_outputs * expert_weights, dim=2
            )  # Weighted sum of experts
            final_output += expert_output

        return final_output


# Example usage
d_model = 512
num_experts = 16
d_ff = 2048
top_k = 2
num_shared_experts = 2

moe_model = DeepSeekMoE(d_model, num_experts, d_ff, top_k, num_shared_experts)
input_tensor = torch.randn(
    10, 15, 512
)  # Batch size of 10, sequence length 15, feature size of 512
output = moe_model(input_tensor)
print(output.shape)  # Should match the input shape
