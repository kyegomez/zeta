import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from typing import Tuple

# Constants
D_MODEL = 1024
D_FF = 4096
NUM_HEADS = 16
NUM_LAYERS = 12
DROPOUT = 0.1
LATENT_DIM = 80
REDUCTION_FACTOR = 4
VOCODER_DIM = 80
KERNEL_SIZE = 5
POST_NET_CHANNELS = 256


# Utility: Log tensor shape
def log_shape(tensor: Tensor, name: str):
    logger.debug(f"{name} shape: {tensor.shape}")
    return tensor


class PreNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super(PreNet, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Reshape input from (batch_size, seq_len, feature_dim) -> (batch_size * seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size * seq_len, feature_dim)
        x = self.net(x)

        # Reshape back to (batch_size, seq_len, output_dim)
        output_dim = x.shape[-1]
        x = x.view(batch_size, seq_len, output_dim)
        return log_shape(x, "PreNet output")


# Transformer-based Autoregressive Language Model (Decoder)
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = D_FF,
        dropout: float = DROPOUT,
    ):
        super(TransformerDecoder, self).__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        out = self.decoder(tgt, memory)
        return log_shape(out, "TransformerDecoder output")


# Latent Sampling Module
class LatentSamplingModule(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super(LatentSamplingModule, self).__init__()
        self.mean_layer = nn.Linear(d_model, LATENT_DIM)
        self.log_var_layer = nn.Linear(d_model, LATENT_DIM)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = log_shape(self.mean_layer(x), "Mean vector")
        log_var = log_shape(self.log_var_layer(x), "Log Variance vector")
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)
        return z, mean, log_var


# Stop Prediction Layer
class StopPredictionLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super(StopPredictionLayer, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        return log_shape(out, "Stop Prediction Layer output")


# PostNet for Mel-Spectrogram Refinement
class PostNet(nn.Module):
    def __init__(
        self,
        input_dim: int = VOCODER_DIM,
        channels: int = POST_NET_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        num_layers: int = 5,
    ):
        super(PostNet, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    input_dim, channels, kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.Tanh())
        layers.append(
            nn.Conv1d(
                channels, input_dim, kernel_size, padding=kernel_size // 2
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = log_shape(x, "PostNet input")
        x = self.net(x.transpose(1, 2)).transpose(
            1, 2
        )  # Conv1D expects (batch_size, channels, time)
        return log_shape(x, "PostNet output")


# MELLE Architecture
class MELLE(nn.Module):
    def __init__(
        self, d_model: int = D_MODEL, reduction_factor: int = REDUCTION_FACTOR
    ):
        super(MELLE, self).__init__()
        self.reduction_factor = reduction_factor

        # PreNets
        self.text_prenet = PreNet(
            input_dim=256, output_dim=d_model
        )  # Example BPE embedding size
        self.mel_prenet = PreNet(input_dim=VOCODER_DIM, output_dim=d_model)

        # Transformer Decoder (Language Model)
        self.decoder = TransformerDecoder(d_model=d_model)

        # Latent Sampling Module
        self.latent_sampling = LatentSamplingModule(d_model=d_model)

        # Stop Prediction Layer
        self.stop_prediction = StopPredictionLayer(d_model=d_model)

        # PostNet
        self.post_net = PostNet(input_dim=VOCODER_DIM)

    def forward(self, text: Tensor, mel: Tensor) -> Tuple[Tensor, Tensor]:
        # Step 1: Pre-process inputs
        text_embeddings = self.text_prenet(text)
        mel_embeddings = self.mel_prenet(mel)

        # Step 2: Autoregressive Generation with Transformer Decoder
        output = self.decoder(mel_embeddings, text_embeddings)

        # Step 3: Latent Sampling
        z, mean, log_var = self.latent_sampling(output)

        # Step 4: Stop Prediction
        stop_logits = self.stop_prediction(output)

        # Step 5: PostNet Refinement
        refined_mel = self.post_net(z)

        return refined_mel, stop_logits


# Loss Functions
def loss_fn(
    y_true: Tensor,
    y_pred: Tensor,
    mean: Tensor,
    log_var: Tensor,
    stop_true: Tensor,
    stop_pred: Tensor,
    lambda_kl: float = 0.1,
    lambda_flux: float = 0.5,
) -> Tensor:
    # Regression Loss (L1 + L2)
    reg_loss = F.l1_loss(y_pred, y_true) + F.mse_loss(y_pred, y_true)

    # KL Divergence Loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # Stop Prediction BCE Loss
    stop_loss = F.binary_cross_entropy_with_logits(stop_pred, stop_true)

    # Spectrogram Flux Loss
    flux_loss = torch.mean(torch.abs(mean[:, 1:] - mean[:, :-1]))

    total_loss = (
        reg_loss + lambda_kl * kl_loss + lambda_flux * flux_loss + stop_loss
    )
    return total_loss


# Example Usage
if __name__ == "__main__":
    # Initialize logger
    logger.add("melle.log", format="{time} {level} {message}", level="DEBUG")

    # Sample inputs (batch_size, seq_len, feature_dim)
    text_input = torch.randn(
        16, 100, 256
    )  # Example: (batch_size, seq_len, BPE_dim)
    mel_input = torch.randn(
        16, 200, VOCODER_DIM
    )  # Example: (batch_size, mel_seq_len, mel_dim)

    # Model instantiation
    model = MELLE()

    # Forward pass
    refined_mel, stop_logits = model(text_input, mel_input)
    logger.debug(f"Refined Mel-Spectrogram shape: {refined_mel.shape}")
    logger.debug(f"Stop Logits shape: {stop_logits.shape}")
