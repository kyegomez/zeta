import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiModalCrossAttention(nn.Module):
    """
    Multi-modal cross attention module for integrating text and image features.

    Args:
    - dim (int): Hidden dimension of the input.
    - num_heads (int): Number of heads for multi-head attention.
    - dropout_rate (float): Dropout probability.
    - normalize_qk (bool): Whether to normalize the query and key vectors.

    Usage:
    - Instantiate the module and pass text and image hidden states to it.
    """

    def __init__(
        self,
        dim,
        num_heads,
        dropout_rate=0.3,
        normalize_qk=True,
        img_size=(32, 32),
        channels=3,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = dim // num_heads
        self.normalize_qk = normalize_qk

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim)

        # Projection layers for text-to-image attention
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # Projection layers for image-to-text attention
        self.query_proj_reverse = nn.Linear(dim, dim)
        self.key_proj_reverse = nn.Linear(dim, dim)
        self.value_proj_reverse = nn.Linear(dim, dim)

        # Output linear layer
        self.output_linear = nn.Linear(2 * dim, dim)

        # Additional layer to match the image feature dimension
        self.image_to_feature_dim = nn.Linear(
            channels * img_size[0] * img_size[1], dim
        )

    def forward(self, text_hidden, image_hidden):
        """
        text_hidden: Hidden states from text model.
        image_hidden: Hidden states from image model (4D tensor).
        """

        # Flatten image features and project to the correct dimension
        image_hidden = rearrange(image_hidden, "b c h w -> b (h w) c")
        image_hidden = self.image_to_feature_dim(image_hidden)

        # Text-to-Image Attention
        query = self.query_proj(text_hidden)
        key = self.key_proj(image_hidden)
        value = self.value_proj(image_hidden)

        if self.normalize_qk:
            query = self.norm(query)
            key = self.norm(key)

        attn_weights = F.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5),
            dim=-1,
        )
        attn_weights = self.dropout(attn_weights)
        text_to_image = torch.matmul(attn_weights, value)

        # Image-to-Text Attention
        query_reverse = self.query_proj_reverse(image_hidden)
        key_reverse = self.key_proj_reverse(text_hidden)
        value_reverse = self.value_proj_reverse(text_hidden)

        if self.normalize_qk:
            query_reverse = self.norm(query_reverse)
            key_reverse = self.norm(key_reverse)

        attn_weights_reverse = F.softmax(
            torch.matmul(query_reverse, key_reverse.transpose(-2, -1))
            / (self.head_dim**0.5),
            dim=-1,
        )
        attn_weights_reverse = self.dropout(attn_weights_reverse)
        image_to_text = torch.matmul(attn_weights_reverse, value_reverse)

        # Concatenate and pass through linear layer
        combined_output = torch.cat((text_to_image, image_to_text), dim=-1)
        output = self.output_linear(combined_output)

        return output

    # Parameters for demonstration


batch_size = 32
text_seq_length = 128
image_height, image_width = 32, 32
channels = 3
feature_dim = 512
num_heads = 8

# Initialize the MultiModalCrossAttention module
cross_attn = MultiModalCrossAttention(
    dim=feature_dim,
    num_heads=num_heads,
    img_size=(image_height, image_width),
    channels=channels,
)

# Generate random text features: [batch_size, text_seq_length, feature_dim]
text_features = torch.randn(batch_size, text_seq_length, feature_dim)

# Generate random image features: [batch_size, channels, image_height, image_width]
image_features = torch.randn(batch_size, channels, image_height, image_width)

# Forward pass
output = cross_attn(text_features, image_features)

# Output shape
print(
    f"Output Shape: {output.shape}"
)  # Expected shape: [batch_size, text_seq_length, feature_dim]
