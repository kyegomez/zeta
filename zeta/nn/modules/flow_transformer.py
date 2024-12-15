import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass
import math
from loguru import logger
from pathlib import Path
from datetime import datetime


@dataclass
class FlowTransformerConfig:
    """Configuration for Flow Transformer.

    Attributes:
        dim: Model dimension
        heads: Number of attention heads
        depth: Number of transformer layers
        seq_length: Maximum sequence length
        flow_hidden_dim: Hidden dimension for flow networks
        flow_steps: Number of flow integration steps
        dropout: Dropout rate
        attention_dropout: Dropout rate for attention
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for learning rate
        max_steps: Maximum number of training steps
        batch_size: Batch size for training
        checkpoint_interval: Number of steps between checkpoints
        device: Device to use for computation
    """

    dim: int = 512
    heads: int = 8
    depth: int = 6
    seq_length: int = 1024
    flow_hidden_dim: int = 128
    flow_steps: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    learning_rate: float = 1e-4
    vocab_size: int = 50000
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 100000
    batch_size: int = 32
    checkpoint_interval: int = 1000
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class FlowMLP(nn.Module):
    def __init__(self, config: FlowTransformerConfig):
        super().__init__()
        self.config = config

        # Flow network for velocity field - keep input dimension as config.dim
        self.flow_net = nn.Sequential(
            nn.Linear(config.dim, config.flow_hidden_dim),
            nn.LayerNorm(config.flow_hidden_dim),
            nn.GELU(),
            nn.Linear(config.flow_hidden_dim, config.flow_hidden_dim),
            nn.LayerNorm(config.flow_hidden_dim),
            nn.GELU(),
            nn.Linear(config.flow_hidden_dim, config.dim),
        )

        # Time embedding
        self.time_embedding = nn.Linear(1, config.dim, bias=False)
        self.time_scale = nn.Parameter(torch.ones(1))

        logger.debug(f"Initialized FlowMLP with architecture: {self.flow_net}")

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, dim = x.shape

        # Initialize flow
        current_state = x

        # Integration steps
        dt = 1.0 / self.config.flow_steps
        for step in range(self.config.flow_steps):
            t = (
                torch.full(
                    (batch_size, seq_length, 1), step * dt, device=x.device
                )
                * self.time_scale
            )

            # Project time to dimension space and add to current state
            t_embedded = self.time_embedding(t)
            flow_input = current_state + t_embedded

            # Compute velocity field
            velocity = self.flow_net(flow_input)

            # Euler integration step
            current_state = current_state + velocity * dt

        return current_state


class MultiHeadAttention(nn.Module):
    def __init__(self, config: FlowTransformerConfig):
        super().__init__()
        self.config = config

        assert config.dim % config.heads == 0, "dim must be divisible by heads"
        self.head_dim = config.dim // config.heads

        self.to_qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.to_out = nn.Linear(config.dim, config.dim)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        nn.init.xavier_normal_(self.to_qkv.weight, gain=0.1)
        nn.init.xavier_normal_(self.to_out.weight, gain=0.1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_length, _ = x.shape

        # Project to q, k, v
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into heads
        q = q.view(
            batch_size, seq_length, self.config.heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_length, self.config.heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_length, self.config.heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, S, S]

        if mask is not None:
            # Expand mask to match attention scores shape
            mask = mask.view(batch_size, 1, 1, seq_length)
            mask = mask.expand(-1, self.config.heads, seq_length, -1)
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        # Apply attention
        attn = F.softmax(scores, dim=-1)
        attn = self.attention_dropout(attn)

        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_length, self.config.dim)

        # Project to output
        out = self.to_out(out)
        out = self.output_dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with FlowMLP."""

    def __init__(self, config: FlowTransformerConfig):
        """Initialize transformer block.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Layer norm for pre-attention and pre-MLP
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)

        # Attention and MLP
        self.attention = MultiHeadAttention(config)
        self.flow_mlp = FlowMLP(config)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Process input through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_length, dim)
        """
        # Attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # FlowMLP with residual
        mlp_out = self.flow_mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)

        return x


class FlowTransformer(nn.Module):
    """Transformer model with FlowMLP blocks."""

    def __init__(self, config: FlowTransformerConfig):
        """Initialize Flow Transformer.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        logger.info(f"Initializing Flow Transformer with config: {config}")

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.seq_length, config.dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.depth)]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.dim)

        # Output projection
        self.to_logits = nn.Linear(config.dim, config.vocab_size)

        self._init_weights()
        self._setup_logging()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"logs/flow_transformer_{timestamp}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="500 MB")

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Process input through Flow Transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_length)
            mask: Optional attention mask

        Returns:
            Output logits of shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.shape

        # Embeddings
        x = self.token_embedding(x)
        x = x + self.position_embedding[:, :seq_length]

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm and projection
        x = self.norm(x)
        logits = self.to_logits(x)

        return logits

    def train_step(
        self, batch: Dict[str, Tensor], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform one training step.

        Args:
            batch: Dictionary containing input_ids and labels
            optimizer: Optimizer instance

        Returns:
            Dictionary of metrics
        """
        optimizer.zero_grad()

        # Forward pass
        logits = self(batch["input_ids"], batch.get("attention_mask"))

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            batch["labels"].view(-1),
            ignore_index=-100,
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return {"loss": loss.item()}

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "FlowTransformer":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded FlowTransformer model
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Model loaded from {path} (saved at {checkpoint['timestamp']})"
        )
        return model


def create_optimizer(
    model: FlowTransformer, config: FlowTransformerConfig
) -> torch.optim.AdamW:
    """Create optimizer with weight decay fix.

    Args:
        model: FlowTransformer model
        config: Configuration object

    Returns:
        AdamW optimizer
    """
    # Separate parameters that should and shouldn't use weight decay
    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or "layer_norm" in fpn:
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(optim_groups, lr=config.learning_rate)


# Create configuration
config = FlowTransformerConfig(
    dim=512,  # Model dimension
    heads=8,  # Number of attention heads
    depth=6,  # Number of transformer layers
    seq_length=1024,  # Maximum sequence length
    vocab_size=50000,  # Vocabulary size
    flow_steps=4,  # Number of flow integration steps
)

# Initialize model
model = FlowTransformer(config)
model = model.to(config.device)

# Create a sample input
batch_size = 4
seq_length = 512
input_ids = torch.randint(
    0, config.vocab_size, (batch_size, seq_length), device=config.device
)

# Optional attention mask (1 for tokens to attend to, 0 for tokens to ignore)
attention_mask = torch.ones((batch_size, seq_length), device=config.device)

# Forward pass
with torch.no_grad():
    output_logits = model(input_ids, attention_mask)

print(f"Input shape: {input_ids.shape}")
print(
    f"Output shape: {output_logits.shape}"
)  # Should be [batch_size, seq_length, vocab_size]

# Get predictions
predictions = torch.argmax(output_logits, dim=-1)
print(
    f"Predictions shape: {predictions.shape}"
)  # Should be [batch_size, seq_length]
