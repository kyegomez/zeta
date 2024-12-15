"""
Evolutionary Transformer Model with Multi-Query Attention and Mixture of Experts

This script implements a transformer model that can reproduce and evolve using an evolutionary algorithm.
The model uses multi-query attention and a mixture of experts in its architecture. The evolutionary
algorithm simulates sexual reproduction through parameter crossover and includes mutation for diversity.

Requirements:
- Python 3.7+
- PyTorch
- loguru

Author: OpenAI ChatGPT
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List

from loguru import logger


class MultiQueryAttention(nn.Module):
    """
    Implements multi-query attention mechanism.
    Instead of having separate keys and values for each head,
    uses shared keys and values across all heads.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initializes the MultiQueryAttention module.

        Args:
            embed_dim (int): Dimension of embedding vector.
            num_heads (int): Number of attention heads.
        """
        super(MultiQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim // num_heads)
        self.value = nn.Linear(embed_dim, embed_dim // num_heads)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-query attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
        """
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # Compute queries
        q = self.query(x)  # Shape: (batch_size, seq_length, embed_dim)

        # Compute keys and values (shared across heads)
        k = self.key(
            x
        )  # Shape: (batch_size, seq_length, embed_dim // num_heads)
        v = self.value(
            x
        )  # Shape: (batch_size, seq_length, embed_dim // num_heads)

        # Split queries into heads
        q = q.view(batch_size, seq_length, self.num_heads, -1)
        k = k.unsqueeze(
            2
        )  # Shape: (batch_size, seq_length, 1, embed_dim // num_heads)
        v = v.unsqueeze(
            2
        )  # Shape: (batch_size, seq_length, 1, embed_dim // num_heads)

        # Compute attention scores
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, k)
        attn_scores = attn_scores / (embed_dim**0.5)

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        attn_output = attn_output.contiguous().view(batch_size, seq_length, -1)

        output = self.out_proj(attn_output)

        return output


class MixtureOfExperts(nn.Module):
    """
    Implements a mixture of experts feed-forward network.
    Uses a gating mechanism to combine outputs from multiple expert networks.
    """

    def __init__(self, embed_dim: int, expert_dim: int, num_experts: int):
        """
        Initializes the MixtureOfExperts module.

        Args:
            embed_dim (int): Dimension of embedding vector.
            expert_dim (int): Dimension of expert network hidden layer.
            num_experts (int): Number of expert networks.
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, expert_dim),
                    nn.ReLU(),
                    nn.Linear(expert_dim, embed_dim),
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mixture of experts.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor after combining expert outputs.
        """
        batch_size, seq_length, embed_dim = x.size()

        # Shape: (batch_size, seq_length, num_experts)
        gate_scores = torch.softmax(self.gate(x), dim=-1)

        # Process each expert separately
        expert_outputs = []
        for expert in self.experts:
            # Shape: (batch_size, seq_length, embed_dim)
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        # Stack along a new dimension
        # Shape: (num_experts, batch_size, seq_length, embed_dim)
        expert_outputs = torch.stack(expert_outputs)

        # Permute to (batch_size, seq_length, embed_dim, num_experts)
        expert_outputs = expert_outputs.permute(1, 2, 3, 0)

        # Reshape gate scores to (batch_size, seq_length, 1, num_experts)
        gate_scores = gate_scores.unsqueeze(2)

        # Weighted sum over experts
        # Shape: (batch_size, seq_length, embed_dim)
        output = torch.sum(expert_outputs * gate_scores, dim=-1)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Defines a single layer of the transformer encoder with multi-query attention and mixture of experts.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, expert_dim: int, num_experts: int
    ):
        """
        Initializes the TransformerEncoderLayer.

        Args:
            embed_dim (int): Dimension of embedding vector.
            num_heads (int): Number of attention heads.
            expert_dim (int): Dimension of expert network hidden layer.
            num_experts (int): Number of expert networks.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiQueryAttention(embed_dim, num_heads)
        self.moe = MixtureOfExperts(embed_dim, expert_dim, num_experts)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        # Self-attention
        src2 = self.self_attn(src)
        src = src + src2
        src = self.norm1(src)

        # Mixture of Experts feed-forward
        src2 = self.moe(src)
        src = src + src2
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """
    Defines the transformer encoder consisting of multiple encoder layers.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        expert_dim: int,
        num_experts: int,
    ):
        """
        Initializes the TransformerEncoder.

        Args:
            num_layers (int): Number of encoder layers.
            embed_dim (int): Dimension of embedding vector.
            num_heads (int): Number of attention heads.
            expert_dim (int): Dimension of expert network hidden layer.
            num_experts (int): Number of expert networks.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim, num_heads, expert_dim, num_experts
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor after encoding.
        """
        output = src
        for layer in self.layers:
            output = layer(output)
        output = self.norm(output)
        return output


class TransformerModel(nn.Module):
    """
    Full transformer model with embedding, encoder, and output layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        expert_dim: int,
        num_experts: int,
    ):
        """
        Initializes the TransformerModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of embedding vector.
            num_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            expert_dim (int): Dimension of expert network hidden layer.
            num_experts (int): Number of expert networks.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = TransformerEncoder(
            num_layers, embed_dim, num_heads, expert_dim, num_experts
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer model.

        Args:
            src (torch.Tensor): Input tensor of token indices (batch_size, seq_length).

        Returns:
            torch.Tensor: Output logits for each token.
        """
        embedded = self.embedding(src)
        encoded = self.encoder(embedded)
        output = self.fc_out(encoded)
        return output


class EvolutionaryAlgorithm:
    """
    Simulates an evolutionary algorithm where transformer models can reproduce and evolve.
    Implements selection, crossover (sexual reproduction), and mutation.
    """

    def __init__(self, population_size: int, model_params: dict):
        """
        Initializes the EvolutionaryAlgorithm.

        Args:
            population_size (int): Number of models in the population.
            model_params (dict): Parameters for initializing transformer models.
        """
        self.population_size = population_size
        self.model_params = model_params
        self.population = [self.create_model() for _ in range(population_size)]
        self.scores = [0.0 for _ in range(population_size)]

    def create_model(self) -> TransformerModel:
        """
        Creates a new transformer model.

        Returns:
            TransformerModel: A new instance of the transformer model.
        """
        model = TransformerModel(**self.model_params)
        return model

    def evaluate_population(self, data_loader: DataLoader) -> None:
        """
        Evaluates the entire population on the provided data.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation dataset.
        """
        self.scores = []
        for idx, model in enumerate(self.population):
            score = self.evaluate_model(model, data_loader)
            self.scores.append(score)
            logger.debug(f"Model {idx} score: {score}")

    def evaluate_model(
        self, model: TransformerModel, data_loader: DataLoader
    ) -> float:
        """
        Evaluates a single model.

        Args:
            model (TransformerModel): The model to evaluate.
            data_loader (DataLoader): DataLoader for the evaluation dataset.

        Returns:
            float: The evaluation score (negative loss).
        """
        model.eval()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)
                )
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        return -avg_loss  # Negative loss as score (higher is better)

    def select_parents(self) -> List[TransformerModel]:
        """
        Selects parent models for reproduction based on their scores.

        Returns:
            List[TransformerModel]: List of selected parent models.
        """
        # Select parents based on scores (e.g., top 50%)
        sorted_population = [
            model
            for _, model in sorted(
                zip(self.scores, self.population),
                key=lambda x: x[0],
                reverse=True,
            )
        ]
        num_parents = self.population_size // 2
        parents = sorted_population[:num_parents]
        return parents

    def crossover(
        self, parent1: TransformerModel, parent2: TransformerModel
    ) -> TransformerModel:
        """
        Performs crossover between two parent models to produce a child model.

        Args:
            parent1 (TransformerModel): The first parent model.
            parent2 (TransformerModel): The second parent model.

        Returns:
            TransformerModel: The resulting child model.
        """
        child = self.create_model()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            mask = torch.rand_like(child_param) > 0.5
            child_param.data.copy_(torch.where(mask, param1.data, param2.data))
        return child

    def mutate(
        self, model: TransformerModel, mutation_rate: float = 0.01
    ) -> None:
        """
        Applies random mutations to a model's parameters.

        Args:
            model (TransformerModel): The model to mutate.
            mutation_rate (float, optional): Probability of mutation per parameter. Defaults to 0.01.
        """
        for param in model.parameters():
            mutation_mask = torch.rand_like(param) < mutation_rate
            param.data[mutation_mask] += (
                torch.randn_like(param.data[mutation_mask]) * 0.1
            )

    def create_next_generation(self) -> None:
        """
        Creates the next generation of the population using parents and offspring.
        """
        parents = self.select_parents()
        next_generation = parents.copy()
        self.population_size - len(parents)
        while len(next_generation) < self.population_size:
            parent_indices = torch.randperm(len(parents))[:2]
            parent1 = parents[parent_indices[0]]
            parent2 = parents[parent_indices[1]]
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            next_generation.append(child)
        self.population = next_generation

    def run_evolution(
        self, data_loader: DataLoader, num_generations: int
    ) -> None:
        """
        Runs the evolutionary algorithm for a specified number of generations.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation dataset.
            num_generations (int): Number of generations to run.
        """
        for generation in range(num_generations):
            logger.info(f"Generation {generation}")
            self.evaluate_population(data_loader)
            best_score = max(self.scores)
            logger.info(f"Best score: {best_score}")
            self.create_next_generation()


def main():
    """
    Main function to set up and run the evolutionary algorithm.
    """
    # Set up logging
    logger.add("evolution.log", rotation="500 MB")

    # Define model parameters
    model_params = {
        "vocab_size": 10000,
        "embed_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "expert_dim": 2048,
        "num_experts": 4,
    }

    # Initialize evolutionary algorithm
    ea = EvolutionaryAlgorithm(population_size=10, model_params=model_params)

    # Prepare data
    # For demonstration, we will use random data
    dummy_input = torch.randint(
        0, model_params["vocab_size"], (32, 50)
    )  # (batch_size, seq_length)
    dummy_target = torch.randint(0, model_params["vocab_size"], (32, 50))
    dataset = torch.utils.data.TensorDataset(dummy_input, dummy_target)
    data_loader = DataLoader(dataset, batch_size=32)

    # Run evolution
    ea.run_evolution(data_loader, num_generations=5)


if __name__ == "__main__":
    main()
