import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from loguru import logger
import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the MoE Transformer model"""

    d_model: int = 512
    n_heads: int = 8
    n_experts: int = 8
    expert_dim: int = 2048
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 512
    vocab_size: int = 30000
    lstm_hidden_size: int = 512
    learning_threshold: float = 0.1
    weight_update_radius: int = 50


class ExpertLayer(nn.Module):
    """Individual expert network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with routing"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.d_model = config.d_model

        # Create experts
        self.experts = nn.ModuleList(
            [
                ExpertLayer(config.d_model, config.expert_dim)
                for _ in range(config.n_experts)
            ]
        )

        # Router network
        self.router = nn.Linear(config.d_model, config.n_experts)

        logger.info(f"Initialized MoE with {config.n_experts} experts")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate routing weights
        route_weights = F.softmax(self.router(x), dim=-1)

        # Initialize output tensor
        final_output = torch.zeros_like(x)

        # Get expert outputs and combine them
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_weight = route_weights[..., i : i + 1]
            final_output += expert_out * expert_weight

        return final_output, route_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads

        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)

        # Linear projections and reshape
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k)

        # Transpose for attention dot product
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attn, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        return self.out(context)


class ContinuousLearningLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.learning_threshold = config.learning_threshold
        self.weight_update_radius = config.weight_update_radius

    def find_relevant_weights(
        self, input_features: torch.Tensor, weight_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Identify weights that are most relevant to the current input"""
        # Flatten both tensors completely
        flat_input = input_features.flatten()
        flat_weights = weight_matrix.flatten()

        # Reshape to 2D for correlation (n x 1)
        input_2d = flat_input.unsqueeze(1)
        weight_2d = flat_weights.unsqueeze(1)

        # Calculate simple correlation
        correlation = torch.abs(input_2d - weight_2d.t())
        mask = (correlation < self.learning_threshold).float()

        # Apply neighborhood effect using 1D max pooling
        mask = F.max_pool1d(
            mask.unsqueeze(0),
            kernel_size=self.weight_update_radius,
            padding=self.weight_update_radius // 2,
            stride=1,
        ).squeeze(0)

        # Reshape back to original shape
        return mask.reshape_as(weight_matrix)

    def apply_selective_updates(
        self, grad: torch.Tensor, relevance_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply gradients only to relevant weights"""
        return grad * relevance_mask.to(grad.device)


class LiquidTransformer(nn.Module):
    """Main model combining transformer with MoE, LSTM, and continuous learning"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.max_seq_length, config.d_model)
        )

        # Main layers
        self.moe = MixtureOfExperts(config)
        self.attention = MultiHeadAttention(config)
        self.lstm = nn.LSTM(
            config.d_model,
            config.lstm_hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Output layers
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Continuous learning component
        self.continuous_learning = ContinuousLearningLayer(config)

        logger.info("Initialized Liquid Transformer model")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Add positional embeddings
        x = self.token_embedding(x) + self.position_embedding[:, : x.size(1)]

        # Self-attention
        attended = self.attention(x, x, x, mask)

        # Mixture of Experts
        moe_output, routing_weights = self.moe(attended)

        # LSTM processing
        lstm_output, (h_n, c_n) = self.lstm(moe_output)

        # Combine and normalize
        output = self.layer_norm(lstm_output + moe_output)
        output = self.dropout(output)

        return {
            "output": output,
            "routing_weights": routing_weights,
            "lstm_state": (h_n, c_n),
        }

    def continuous_update(self, loss: torch.Tensor):
        """Perform continuous learning update"""
        # Get gradients
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)

        for param, grad in zip(self.parameters(), grads):
            if grad is not None:
                # Find relevant weights for this input
                relevance_mask = self.continuous_learning.find_relevant_weights(
                    param.data, grad
                )

                # Apply selective updates
                selective_grad = (
                    self.continuous_learning.apply_selective_updates(
                        grad, relevance_mask
                    )
                )

                # Update weights
                param.data -= self.config.learning_threshold * selective_grad


def create_model(config: Optional[ModelConfig] = None) -> LiquidTransformer:
    """Factory function to create a new model instance"""
    if config is None:
        config = ModelConfig()

    model = LiquidTransformer(config)
    logger.info(f"Created model with config: {config}")
    return model


# Example usage:
if __name__ == "__main__":
    logger.info("Initializing model...")

    # Create configuration
    config = ModelConfig(
        d_model=512, n_heads=8, n_experts=8, expert_dim=2048, vocab_size=30000
    )

    # Initialize model
    model = create_model(config)

    # Example forward pass
    batch_size, seq_length = 32, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    outputs = model(input_ids)
    logger.info(f"Model output shape: {outputs['output'].shape}")

    # Example continuous learning update
    loss = outputs["output"].mean()
    model.continuous_update(loss)
    logger.info("Completed continuous learning update")
