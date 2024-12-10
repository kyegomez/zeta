from typing import List

import torch
from torch import Tensor, nn


class MMLayerNorm(nn.Module):
    def __init__(self, num_modalities: int, dim, epsilon: float = 1e-5):
        """
        Multi-Modality Layer Normalization module.

        Args:
            num_modalities (int): Number of modalities to be fused.
            dim (int): Dimension of the input tensors.
            epsilon (float, optional): Small value added to the denominator for numerical stability. Defaults to 1e-5.

        Examples:
            >>> from zeta.nn.modules import MMLayerNorm
            >>> import torch
            >>> mm_ln = MMLayerNorm(num_modalities=2, dim=64)
            >>> modality1 = torch.randn(32, 10, 64)
            >>> modality2 = torch.randn(32, 10, 64)
            >>> output = mm_ln([modality1, modality2])
            >>> output.shape
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.dim = dim
        self.epsilon = epsilon

        # Learnable weights for fusing modalities
        self.fusion_weights = nn.Parameter(torch.ones(num_modalities))

        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, modalities: List[Tensor]):
        """
        Forward pass of the MMLayerNorm module.

        Args:
            modalities (List[Tensor]): List of input tensors representing different modalities.

        Returns:
            Tensor: Output tensor after fusing and normalizing the modalities.
        """
        assert all(
            [modality.shape == modalities[0].shape for modality in modalities]
        ), "All modalities must have the same shape."

        normalized_modalities = []

        for modality, weight in zip(modalities, self.fusion_weights):
            mean = modality.mean(dim=(1, 2), keepdim=True)
            std = modality.std(dim=(1, 2), keepdim=True)
            normalized = (modality - mean) / (std + self.epsilon)
            weighted_normalized = weight * normalized
            normalized_modalities.append(weighted_normalized)

        # Combine all modalities
        combined = sum(normalized_modalities)

        # Apply learnable scale and shift
        output = self.gamma * combined + self.beta
        return output
