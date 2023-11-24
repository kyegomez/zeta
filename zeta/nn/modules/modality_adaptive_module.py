import torch
from torch import nn
import torch.nn.functional as F
from zeta.nn.attention import FlashAttention


class ModalityAdaptiveModule(nn.Module):
    """
    Modality Adaptive Module

    Args:
        dim: int
            The dimension of the input features
        heads: int
            The number of heads to use for the attention mechanism

    Returns:
        x: torch.Tensor


    Examples:
        >>> x = torch.randn(1, 3, 512)
        >>> y = torch.randn(1, 3, 512)
        >>> model = ModalityAdaptiveModule(512, 8)
        >>> out = model(x, y)
        >>> print(out.shape)
        torch.Size([1, 3, 512])


    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super(ModalityAdaptiveModule, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.scale = dim**-0.5
        assert dim % heads == 0, f"dim must alwasy be divisible by heads"

        # Initialize the normalization layers for each modality
        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        # Initialize the img linear layers
        self.img_v_proj = nn.Linear(dim, dim)
        self.img_k_proj = nn.Linear(dim, dim)

        # Initialize the linear layers for the text
        self.text_v_proj = nn.Linear(dim, dim)
        self.text_k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)

        # Initialize the linear layer
        self.proj = nn.Linear(dim, dim)

        # Attention
        self.attn = FlashAttention(causal=True, dropout=dropout, flash=False)

    def modality_indicator(self, x):
        """Function that returns the modality indicator"""
        if x.dim() == 4:
            return 0
        elif x.dim() == 3:
            return 1
        else:
            raise ValueError("The tensor must be 3 or 4 dimensions")

        # indicator = nn.Linear(self.dim, self.heads)
        # modality_weights = torch.sigmoid(indicator(x))
        # return modality_weights

    # def forward(self, text, img):
    #     """Forward pass of the modality adaptive module"""

    #     # Normalize the text and image features
    #     text_normalized = self.norm_text(text)
    #     img_normalized = self.norm_img(img)

    #     # Concatenate the normalized text and image features
    #     norms_concat = torch.concat((text_normalized, img_normalized))

    #     # Project the text and image features to the same dimension
    #     vision_v = self.img_v_proj(img_normalized)
    #     vision_k = self.img_k_proj(img_normalized)
    #     # Text features are projected to the same dimension as the image features
    #     text_v = self.text_v_proj(text_normalized)
    #     text_k = self.text_k_proj(text_normalized)

    #     # Combine keys from both modalities
    #     k = torch.cat((text_k, vision_k))
    #     v = torch.cat((text_v, vision_v))

    #     # # Project the query to the same dimension as the image and text features
    #     q = self.q_proj(norms_concat)

    #     # # Matmul between the query and the keys
    #     # matmuled = torch.matmul(q, keys_combined)

    #     # # add scale
    #     # matmul_scale = matmuled * self.scale

    #     # # Attention mechanism: dot product of queries and keys, scaled and normalized
    #     # attn = torch.softmax(matmul_scale)

    #     # # Matmul between the softmaxed matmuled and the values
    #     # x = torch.matmul(attn, values_combined)

    #     attn = self.attn(q, k, v)

    #     # Projected matmul
    #     x = self.proj(attn)

    #     # Normalize the outputs
    #     normed_text = self.norm_text(x)
    #     normed_img = self.norm_img(x)
    #     x = torch.concat((normed_text, normed_img))

    #     return x

    def forward(self, text, img):
        batch_size = text.size(0)

        # Normalize the text and image features
        text_normalized = self.norm_text(text)
        img_normalized = self.norm_img(img)

        # Project the text and image features to the same dimension
        vision_v = self.img_v_proj(img_normalized).view(
            batch_size, -1, self.heads, self.dim // self.heads
        )
        vision_k = self.img_k_proj(img_normalized).view(
            batch_size, -1, self.heads, self.dim // self.heads
        )
        text_v = self.text_v_proj(text_normalized).view(
            batch_size, -1, self.heads, self.dim // self.heads
        )
        text_k = self.text_k_proj(text_normalized).view(
            batch_size, -1, self.heads, self.dim // self.heads
        )

        # Combine keys and values from both modalities
        keys_combined = torch.cat((text_k, vision_k), dim=1)
        values_combined = torch.cat((text_v, vision_v), dim=1)

        # Project the query to the same dimension as the image and text features
        queries = self.q_proj(
            torch.cat((text_normalized, img_normalized), dim=1)
        )
        queries = queries.view(
            batch_size, -1, self.heads, self.dim // self.heads
        )

        # Compute the scaled dot-product attention
        # (batch_size, heads, seq_len_q, seq_len_k)
        attention_scores = torch.einsum(
            "bhid,bhjd->bhij", queries, keys_combined
        )
        attention_scores = attention_scores * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the attention to the values
        # (batch_size, heads, seq_len_q, depth_v)
        attention_output = torch.einsum(
            "bhij,bhjd->bhid", attention_weights, values_combined
        )

        # Concatenate the heads
        attention_output = attention_output.contiguous().view(
            batch_size, -1, self.dim
        )

        # Apply dropout if necessary
        attention_output = F.dropout(
            attention_output, p=self.dropout, training=self.training
        )

        # Project the output of the attention mechanism
        x = self.proj(attention_output)

        # Normalize the outputs
        normed_text = self.norm_text(x)
        normed_img = self.norm_img(x)
        x = normed_text + normed_img

        return x


x = torch.randn(1, 3, 512)
y = torch.randn(1, 3, 512)

model = ModalityAdaptiveModule(512, 8)

out = model(x, y)

print(out.shape)
