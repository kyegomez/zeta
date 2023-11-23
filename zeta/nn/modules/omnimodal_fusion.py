import torch
import torch.nn as nn


class OmniModalFusion(nn.Module):
    """
    OmniModalFusion is designed to fuse an arbitrary number of modalities with unknown shapes.

    Args:
        fusion_dim (int): The size of the common embedding space where all modalities are fused.

    Attributes:

    Forward Args:
        *modalities (List[torch.Tensor]): A variable number of tensors, each representing a different modality.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, fusion_dim] representing the fused embeddings.
    """

    def __init__(
        self,
        fusion_dim: int,
    ):
        super(OmniModalFusion, self).__init__()
        self.fusion_dim = fusion_dim
        self.modality_encoders = (
            nn.ModuleList()
        )  # List to hold encoders for each modality

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        # Dynamically add encoders for new modalities
        while len(self.modality_encoders) < len(modalities):
            input_dim = modalities[
                len(self.modality_encoders)
            ].nelement() // modalities[len(self.modality_encoders)].size(
                0
            )  # Compute flattened input dimension
            self.modality_encoders.append(nn.Linear(input_dim, self.fusion_dim))

        embeddings = []
        for i, modality in enumerate(modalities):
            flattened = modality.view(modality.size(0), -1)  # Flatten modality
            embeddings.append(self.modality_encoders[i](flattened))

        # Stack embeddings
        stacked_embeddings = torch.stack(
            embeddings, dim=1
        )  # [batch_size, num_modalities, fusion_dim]

        # Compute attention scores
        attention_scores = torch.mean(
            stacked_embeddings, dim=-1
        )  # Averaging method for simplicity
        attention_weights = torch.nn.functional.softmax(
            attention_scores, dim=1
        ).unsqueeze(-1)

        # Compute weighted fusion
        fused = torch.sum(attention_weights * stacked_embeddings, dim=1)

        return fused


# # Usage Test
# if __name__ == "__main__":
#     fusion_dim = 512
#     model = OmniModalFusion(fusion_dim)

#     batch_size = 16

#     # Simulating 3 modalities with unknown shapes
#     modality1 = torch.rand(
#         batch_size, 10, 5, 512
#     )  # Example: Text [batch_size, seq_len, features]
#     modality2 = torch.rand(
#         batch_size, 64, 64, 3
#     )  # Example: Image [batch_size, height, width, channels]
#     # modality3 = torch.rand(
#     #     batch_size, 4, 32, 32, 1024
#     # )  # Example: 3D Scene [batch_size, depth, height, width, features]

#     modality5 = torch.rand(batch_size, 4, 32, 32, 1024, 244)

#     fused = model(modality1, modality2)
#     print(f"Fused output shape: {fused.shape}")  # Expected: [batch_size, fusion_dim]
