import torch
import torch.nn as nn
from typing import List


class MIRoPE(nn.Module):
    def __init__(self, dim: int):
        """
        Initializes the MI-RoPE module.

        Args:
            dim (int): The dimension of the model's hidden states.
        """
        super(MIRoPE, self).__init__()
        self.dim = dim

    def forward(
        self,
        visual_features: List[torch.Tensor],
        sequence_positions: List[int],
        max_seq_len: int,
    ) -> List[torch.Tensor]:
        """
        Applies the Multimodal-Interleaved Rotary Position Embedding to visual features.

        Args:
            visual_features (List[torch.Tensor]): A list of tensors containing the visual features for each image.
            sequence_positions (List[int]): The positions of the images in the interleaved sequence.
            max_seq_len (int): The maximum sequence length for the rotary position embedding.

        Returns:
            List[torch.Tensor]: The visual features with applied rotary position embeddings.
        """
        assert len(visual_features) == len(
            sequence_positions
        ), "Each image must have a corresponding position."

        # Generate the rotary position embedding
        position_ids = torch.arange(
            0, max_seq_len, dtype=torch.float
        ).unsqueeze(1)
        half_dim = self.dim // 2

        # Correct calculation for the embeddings
        emb = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float)
            * -(torch.log(torch.tensor(10000.0)) / half_dim)
        )
        sin_emb = torch.sin(position_ids * emb)
        cos_emb = torch.cos(position_ids * emb)

        # Concatenate sin and cos embeddings properly
        rotary_emb = torch.cat(
            [sin_emb, cos_emb], dim=1
        )  # This should have shape [max_seq_len, dim]

        embedded_visuals = []
        for i, visual in enumerate(visual_features):
            position = sequence_positions[i]
            # Apply the rotary position embedding based on the sequence position of the image
            visual = self.apply_rotary_embedding(visual, rotary_emb[position])
            embedded_visuals.append(visual)

        return embedded_visuals

    def apply_rotary_embedding(
        self, visual: torch.Tensor, rotary_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the rotary position embedding to a visual feature.

        Args:
            visual (torch.Tensor): The visual feature tensor of shape (num_patches, dim).
            rotary_embedding (torch.Tensor): The rotary embedding corresponding to the position in the sequence.

        Returns:
            torch.Tensor: The visual feature tensor with rotary position embedding applied.
        """
        return (visual * rotary_embedding.cos()) + (
            torch.roll(visual, shifts=1, dims=-1) * rotary_embedding.sin()
        )


# # Assuming batch size of 1 for simplicity, you can generalize this as needed
# batch_size = 1
# dim = 512
# max_seq_len = 100

# # Example inputs
# visual_features = [
#     torch.rand(batch_size, 10, dim),
#     torch.rand(batch_size, 10, dim),
# ]  # 10 patches per image
# sequence_positions = [5, 10]  # Image positions in the interleaved sequence

# # Initialize modules
# mi_rope = MIRoPE(dim=dim)
# # Apply MI-RoPE to each image's visual features
# embedded_visuals = mi_rope(visual_features, sequence_positions, max_seq_len)
# print(embedded_visuals[0].shape)  # Output shape: torch.Size([1, 10, 512])
