import torch
import torchaudio
from PIL import Image
import torchvision.transforms as T
from loguru import logger
from typing import Optional, Dict
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from dataclasses import dataclass
import timm
from transformers import SiglipModel
import faiss
import math
from typing import Any
import numpy as np
import logging

# Configure logger
logger.add("model.log", rotation="500 MB")


@dataclass
class ModelConfig:
    """Configuration for the advanced model architecture."""

    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 12
    num_experts: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 2048
    embedding_dim: int = 512
    num_memory_slots: int = 1000000


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention implementation for efficient processing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, self.head_dim)
        self.v_proj = nn.Linear(config.d_model, self.head_dim)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim)

        # Expand k and v for all heads
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        return self.o_proj(out.reshape(batch_size, seq_len, -1))


class SelfHealingModule(nn.Module):
    """
    Enhanced self-healing module with comprehensive monitoring and adaptive healing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Health metrics tracking
        self.health_metrics = {
            "attention_scores": [],
            "gradient_norms": [],
            "activation_stats": [],
            "memory_usage": [],
            "anomaly_scores": [],
        }

        # Anomaly detection thresholds
        self.thresholds = {
            "gradient_norm": 10.0,
            "attention_entropy": 0.1,
            "activation_range": (-5.0, 5.0),
            "memory_usage": 0.9,  # 90% of available memory
        }

        # Recovery mechanisms
        self.gradient_clipper = nn.utils.clip_grad_norm_
        self.activation_stabilizer = nn.LayerNorm(config.d_model)
        self.attention_regularizer = nn.Dropout(p=0.1)

        # Adaptive components
        self.adaptive_learning_rate = nn.Parameter(torch.ones(1))
        self.health_score = nn.Parameter(torch.ones(1))

        # Memory management
        self.memory_buffer = []
        self.max_memory_entries = 1000

        # Logging
        self.logger = logging.getLogger(__name__)

    def check_attention_health(self, attention_scores: Tensor) -> float:
        """Monitor attention pattern health using entropy."""
        entropy = (
            -(attention_scores * torch.log(attention_scores + 1e-9))
            .sum(-1)
            .mean()
        )
        return entropy.item()

    def check_gradient_health(self, model: nn.Module) -> float:
        """Monitor gradient norms across model parameters."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def check_activation_health(self, activations: Tensor) -> Dict[str, float]:
        """Monitor activation statistics."""
        return {
            "mean": activations.mean().item(),
            "std": activations.std().item(),
            "max": activations.max().item(),
            "min": activations.min().item(),
        }

    def check_memory_health(self) -> float:
        """Monitor memory usage."""
        if torch.cuda.is_available():
            return (
                torch.cuda.memory_allocated()
                / torch.cuda.max_memory_allocated()
            )
        return 0.0

    def detect_anomalies(self, x: Tensor) -> bool:
        """Detect anomalies in input tensor."""
        return torch.isnan(x).any() or torch.isinf(x).any()

    def heal_gradients(self, model: nn.Module):
        """Apply gradient healing mechanisms."""
        if self.check_gradient_health(model) > self.thresholds["gradient_norm"]:
            self.gradient_clipper(
                model.parameters(), self.thresholds["gradient_norm"]
            )
            self.adaptive_learning_rate.data *= 0.9
            self.logger.warning(
                "Gradient instability detected - applying healing"
            )

    def heal_activations(self, x: Tensor) -> Tensor:
        """Stabilize activations."""
        if x.abs().max() > self.thresholds["activation_range"][1]:
            x = self.activation_stabilizer(x)
            self.logger.warning(
                "Activation instability detected - applying normalization"
            )
        return x

    def heal_attention(self, attention_scores: Tensor) -> Tensor:
        """Regularize attention patterns."""
        if (
            self.check_attention_health(attention_scores)
            < self.thresholds["attention_entropy"]
        ):
            attention_scores = self.attention_regularizer(attention_scores)
            self.logger.warning(
                "Attention entropy too low - applying regularization"
            )
        return attention_scores

    def manage_memory(self):
        """Manage memory usage and cleanup."""
        if len(self.memory_buffer) > self.max_memory_entries:
            self.memory_buffer = self.memory_buffer[-self.max_memory_entries :]
            torch.cuda.empty_cache()
            self.logger.info("Memory cleanup performed")

    def update_health_metrics(self, model: nn.Module, x: Tensor):
        """Update health monitoring metrics."""
        self.health_metrics["gradient_norms"].append(
            self.check_gradient_health(model)
        )
        self.health_metrics["activation_stats"].append(
            self.check_activation_health(x)
        )
        self.health_metrics["memory_usage"].append(self.check_memory_health())

        # Update overall health score
        recent_metrics = {
            "gradient": np.mean(self.health_metrics["gradient_norms"][-10:]),
            "activation": np.mean(
                [
                    s["std"]
                    for s in self.health_metrics["activation_stats"][-10:]
                ]
            ),
            "memory": np.mean(self.health_metrics["memory_usage"][-10:]),
        }

        health_score = (
            1.0
            - (
                (recent_metrics["gradient"] / self.thresholds["gradient_norm"])
                + (recent_metrics["memory"] / self.thresholds["memory_usage"])
            )
            / 2
        )

        self.health_score.data = torch.tensor(
            [max(0.0, min(1.0, health_score))]
        )

    def forward(self, x: Tensor, model: Optional[nn.Module] = None) -> Tensor:
        """
        Apply self-healing mechanisms during forward pass.
        """
        # Check for anomalies
        if self.detect_anomalies(x):
            self.logger.error("Anomalies detected in input tensor")
            x = torch.nan_to_num(
                x,
                nan=0.0,
                posinf=self.thresholds["activation_range"][1],
                neginf=self.thresholds["activation_range"][0],
            )

        # Apply healing mechanisms
        x = self.heal_activations(x)

        if model is not None:
            self.heal_gradients(model)
            self.update_health_metrics(model, x)

        # Manage memory
        self.manage_memory()

        # Log health status
        if self.health_score.item() < 0.5:
            self.logger.warning(
                f"Low health score: {self.health_score.item():.2f}"
            )

        return x

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        return {
            "health_score": self.health_score.item(),
            "gradient_norm": np.mean(
                self.health_metrics["gradient_norms"][-10:]
            ),
            "activation_stats": {
                k: np.mean(
                    [
                        s[k]
                        for s in self.health_metrics["activation_stats"][-10:]
                    ]
                )
                for k in ["mean", "std", "max", "min"]
            },
            "memory_usage": np.mean(self.health_metrics["memory_usage"][-10:]),
            "learning_rate_adjustment": self.adaptive_learning_rate.item(),
        }


class ExpertLayer(nn.Module):
    """
    Single expert in the Mixture of Experts system.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ff1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.ff2 = nn.Linear(4 * config.d_model, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.ff2(self.act(self.ff1(x))))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts implementation with routing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList(
            [ExpertLayer(config) for _ in range(config.num_experts)]
        )
        self.router = nn.Linear(config.d_model, config.num_experts)

    def forward(self, x: Tensor) -> Tensor:
        # Compute routing weights
        routing_weights = F.softmax(self.router(x), dim=-1)

        # Initialize output tensor
        final_output = torch.zeros_like(x)

        # Route input to experts
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            final_output += routing_weights[..., i : i + 1] * expert_output

        return final_output


class MemorySystem(nn.Module):
    """
    Long-term memory system using FAISS for efficient similarity search.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.index = faiss.IndexFlatL2(config.embedding_dim)
        self.memory_values = []

    def store(self, embeddings: Tensor, values: List[dict]):
        """Store new embeddings and their associated values."""
        embeddings_np = embeddings.detach().cpu().numpy()
        self.index.add(embeddings_np)
        self.memory_values.extend(values)

    def query(
        self, query_embedding: Tensor, k: int = 5
    ) -> Tuple[Tensor, List[dict]]:
        """Query the memory system for similar embeddings."""
        query_np = query_embedding.detach().cpu().numpy()
        distances, indices = self.index.search(query_np, k)
        retrieved_values = [self.memory_values[i] for i in indices[0]]
        return torch.from_numpy(distances), retrieved_values


class PerceptionModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Vision encoder (using a stable timm model)
        self.vision_encoder = timm.create_model(
            "vit_large_patch16_384",  # More stable choice
            pretrained=True,
            num_classes=0,  # Remove classification head
        )

        # SIGLIP vision model only
        self.siglip = SiglipModel.from_pretrained(
            "google/siglip-base-patch16-384"
        )
        self.siglip_vision = (
            self.siglip.vision_model
        )  # Use only the vision part

        # Audio encoder (using torchaudio)
        self.audio_encoder = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

        # Add projection layers for each visual feature type
        self.vit_proj = nn.Linear(
            1024, config.d_model
        )  # ViT features projection
        self.siglip_proj = nn.Linear(
            768, config.d_model
        )  # SIGLIP features projection
        self.audio_proj = nn.Linear(768, config.d_model)  # wav2vec2 output dim

    def forward(
        self, images: Optional[Tensor] = None, audio: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        outputs = {}

        if images is not None:
            # Process images through vision encoder
            with torch.no_grad():
                vision_features = self.vision_encoder(images)
                # Process through SIGLIP vision model only
                siglip_vision_output = self.siglip_vision(pixel_values=images)
                siglip_features = siglip_vision_output.last_hidden_state.mean(
                    dim=1
                )

            # Project each feature type to common dimension
            vision_features = self.vit_proj(vision_features)
            siglip_features = self.siglip_proj(siglip_features)

            # Combine visual features after projection
            vision_output = vision_features + siglip_features
            outputs["vision"] = vision_output

        if audio is not None:
            # Process audio through wav2vec2
            with torch.no_grad():
                audio_features, _ = self.audio_encoder(audio)

            # Project audio features
            audio_output = self.audio_proj(audio_features.mean(dim=1))
            outputs["audio"] = audio_output

        return outputs


class AdvancedModel(nn.Module):
    """
    Main model class integrating all components.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Core components
        self.perception = PerceptionModule(config)
        self.memory = MemorySystem(config)

        # Transformer layers with MQA and MoE
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": MultiQueryAttention(config),
                        "moe": MixtureOfExperts(config),
                        "norm1": nn.LayerNorm(config.d_model),
                        "norm2": nn.LayerNorm(config.d_model),
                    }
                )
                for _ in range(config.num_layers)
            ]
        )

        # Real-time learning components
        self.learning_rate_adjustment = nn.Parameter(torch.ones(1))
        self.online_adaptation_temp = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        images: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
        memory_query: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Optional tensor of images [batch_size, channels, height, width]
            audio: Optional tensor of audio signals [batch_size, sequence_length]
            memory_query: Optional tensor for querying memory system

        Returns:
            Dictionary containing model outputs and intermediate representations
        """
        # Process inputs through perception module
        perceptual_outputs = self.perception(images, audio)

        if not perceptual_outputs:  # If no inputs were provided
            raise ValueError(
                "At least one input modality (images or audio) must be provided"
            )

        # Query memory system if requested
        if memory_query is not None:
            memory_distances, memory_values = self.memory.query(memory_query)
            perceptual_outputs["memory"] = {
                "distances": memory_distances,
                "values": memory_values,
            }

        # Combine all modalities that are present
        available_features = [
            tensor
            for tensor in perceptual_outputs.values()
            if isinstance(tensor, Tensor)
        ]

        if not available_features:
            raise ValueError("No valid features were extracted from the inputs")

        combined_features = torch.stack(available_features, dim=1)

        # Process through transformer layers
        x = combined_features
        for layer in self.layers:
            # Multi-Query Attention
            attn_out = layer["attention"](layer["norm1"](x))
            x = x + attn_out

            # Mixture of Experts
            moe_out = layer["moe"](layer["norm2"](x))
            x = x + moe_out

            # Apply real-time learning adjustment
            x = x * (
                1 + self.learning_rate_adjustment * self.online_adaptation_temp
            )

        return {"output": x, "perceptual_outputs": perceptual_outputs}

    def self_heal(self):
        """
        Implement self-healing mechanism to detect and correct potential issues.
        """
        logger.info("Performing self-healing checks...")

        # Check and adjust learning rate
        if self.learning_rate_adjustment.item() < 0.1:
            logger.warning("Learning rate adjustment too low, resetting...")
            with torch.no_grad():
                self.learning_rate_adjustment.fill_(1.0)

        # Check memory system health
        if len(self.memory.memory_values) > self.config.num_memory_slots:
            logger.warning("Memory system full, pruning oldest entries...")
            self.memory.memory_values = self.memory.memory_values[
                -self.config.num_memory_slots :
            ]

        # Log system status
        logger.info(
            f"Current model state: LR adjustment={self.learning_rate_adjustment.item():.3f}, "
            f"Memory size={len(self.memory.memory_values)}"
        )


# Example usage
def create_model() -> AdvancedModel:
    """Create and initialize the model with default configuration."""
    config = ModelConfig()
    model = AdvancedModel(config)
    logger.info("Model initialized successfully")
    return model


def prepare_inputs(
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for the model from image and audio files.
    """
    inputs = {}

    # Image preprocessing
    if image_path:
        # Transform for ViT model which expects 384x384
        transform = T.Compose(
            [
                T.Resize(384),  # Changed from 448 to 384
                T.CenterCrop(384),  # Changed from 448 to 384
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        inputs["images"] = image_tensor.to(device)

    # Audio preprocessing remains the same
    if audio_path:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs["audio"] = waveform.unsqueeze(0).to(device)

    return inputs


def run_inference(
    model: AdvancedModel,
    inputs: Dict[str, torch.Tensor],
    memory_query: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run inference with the model.

    Args:
        model: Instance of AdvancedModel
        inputs: Dictionary of input tensors
        memory_query: Optional tensor for memory system query

    Returns:
        Dictionary containing model outputs
    """
    logger.info("Starting inference...")

    try:
        # Set model to evaluation mode
        model.eval()

        with torch.no_grad():
            # Forward pass
            outputs = model(
                images=inputs.get("images"),
                audio=inputs.get("audio"),
                memory_query=memory_query,
            )

            logger.info("Inference completed successfully")
            return outputs

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


# Example usage
def main():
    # Create model
    logger.info("Initializing model...")
    config = ModelConfig()
    model = create_model()

    # Move model to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Example 1: Image-only inference
    logger.info("Running image-only inference...")
    image_inputs = prepare_inputs(image_path="image.png", device=device)
    image_outputs = run_inference(model, image_inputs)
    print(image_outputs)

    # # Example 2: Audio-only inference
    # logger.info("Running audio-only inference...")
    # audio_inputs = prepare_inputs(
    #     audio_path="example_audio.wav",
    #     device=device
    # )
    # audio_outputs = run_inference(model, audio_inputs)

    # # Example 3: Multi-modal inference with memory query
    # logger.info("Running multi-modal inference with memory query...")
    # multimodal_inputs = prepare_inputs(
    #     image_path="example_image.jpg",
    #     audio_path="example_audio.wav",
    #     device=device
    # )

    # # Create a sample memory query (e.g., from previous embedding)
    # memory_query = torch.randn(1, config.embedding_dim).to(device)

    # multimodal_outputs = run_inference(
    #     model,
    #     multimodal_inputs,
    #     memory_query=memory_query
    # )

    # # Process outputs
    # for output_type, tensor in multimodal_outputs['perceptual_outputs'].items():
    #     if isinstance(tensor, torch.Tensor):
    #         logger.info(f"{output_type} output shape: {tensor.shape}")

    # # Access memory results if available
    # if 'memory' in multimodal_outputs['perceptual_outputs']:
    #     memory_results = multimodal_outputs['perceptual_outputs']['memory']
    #     logger.info(f"Retrieved {len(memory_results['values'])} memory entries")

    # # Access final output
    # final_output = multimodal_outputs['output']
    # logger.info(f"Final output shape: {final_output.shape}")

    # Optionally run self-healing
    model.self_heal()


if __name__ == "__main__":
    main()
