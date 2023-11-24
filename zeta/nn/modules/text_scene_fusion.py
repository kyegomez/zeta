import torch
import torch.nn as nn
from einops import repeat


class TextSceneAttentionFusion(nn.Module):
    """
    TextSceneAttentionFusion is an attention-based fusion mechanism to combine
    text sequences and 3D scene embeddings. The model computes attention scores
    for each text token with respect to the 3D scene to focus on specific parts
    of the text that are more relevant to the scene.

    Args:
        text_features (int): Dimension of the text embeddings.
        scene_features (int): Dimension of the scene embeddings.

    Forward Args:
        text (torch.Tensor): Tensor of shape [batch_size, seq_len, text_features]
                             representing sequences of text embeddings.
        scene (torch.Tensor): Tensor of shape [batch_size, depth, height, width, scene_features]
                              representing 3D scene embeddings.

    Returns:
        torch.Tensor: Tensor of shape [batch_size, seq_len, scene_features]
                      representing the fused embeddings.
    """

    def __init__(self, text_features: int, scene_features: int):
        super(TextSceneAttentionFusion, self).__init__()

        # A linear layer for calculating attention scores
        self.attention = nn.Linear(text_features + scene_features, 1)

    def forward(self, text: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        # Flattening spatial dimensions of the scene for simplicity
        batch_size, depth, height, width, scene_features = scene.shape
        scene_flat = scene.view(
            batch_size, depth * height * width, scene_features
        )

        # Using einops to repeat the scene tensor for matching text sequence length
        scene_expanded = repeat(
            scene_flat, "b sh sf -> b st sh sf", st=text.size(1)
        )

        # Repeating the text tensor to match the flattened spatial dimensions of the scene
        text_expanded = repeat(
            text, "b st tf -> b st sh tf", sh=depth * height * width
        )

        # Concatenating expanded scene tensor and text tensor
        concat_features = torch.cat(
            [text_expanded, scene_expanded], dim=-1
        )  # [batch_size, seq_len, depth*height*width, text_features + scene_features]

        # Computing attention scores and weights
        attention_scores = self.attention(concat_features).squeeze(
            -1
        )  # [batch_size, seq_len, depth*height*width]
        attention_weights = torch.nn.functional.softmax(
            attention_scores.flatten(2), dim=-1
        ).view(batch_size, seq_len, depth * height * width, 1)

        # Using einsum to obtain weighted scene embeddings
        fused = torch.einsum(
            "btsh,btshj->btsj", attention_weights, scene_expanded
        )

        return fused


# Usage Test
if __name__ == "__main__":
    text_features = 512
    scene_features = 1024
    model = TextSceneAttentionFusion(text_features, scene_features)

    batch_size = 16
    seq_len = 10
    depth = 4
    height = 64
    width = 64

    text = torch.rand(batch_size, seq_len, text_features)
    scene = torch.rand(batch_size, depth, height, width, scene_features)

    fused = model(text, scene)
    print(
        f"Fused output shape: {fused.shape}"
    )  # Expected: [batch_size, seq_len, scene_features]
