"""
DOES NOT WORK:
    - Need to configure the input shape to match the input shape of regular text features

VisuaL Expert module from: https://arxiv.org/pdf/2311.03079.pdf

Visual expert module. We add a visual expert module to each layer to enable deep visual-language
feature alignment. Specifically, the visual expert module in each layer consists of a QKV matrix
and an MLP in each layer. The shapes of the QKV matrix and MLP are identical to those in the
pretrained language model and initialized from them. The motivation is that each attention head
in the language model captures a certain aspect of semantic information, while a trainable visual
expert can transform the image features to align with the different heads, therefore enabling deep
fusion.

Formally, suppose that the input hidden states of an attention layer are X ∈ R
B×H×(LI+LT )×D,
where B is the batch size, LI and LT are the lengths of image and text sequences, H is the number
of attention heads, and D is the hidden size. In the attention with visual expert, X is first split as
4

Shape = B, SEQ_LEN, DIM or regular text shape
"""
import torch
from torch import nn

from zeta.nn.attention.multihead_attention import MultiheadAttention
from zeta.nn.modules.simple_feedforward import SimpleFeedForward


class VisualExpert:
    """
    Visual Expert from https://arxiv.org/pdf/2311.03079.pdf

    Visual expert module. We add a visual expert module to each layer to enable deep visual-language
    feature alignment. Specifically, the visual expert module in each layer consists of a QKV matrix
    and an MLP in each layer. The shapes of the QKV matrix and MLP are identical to those in the
    pretrained language model and initialized from them. The motivation is that each attention head
    in the language model captures a certain aspect of semantic information, while a trainable visual
    expert can transform the image features to align with the different heads, therefore enabling deep
    fusion.

    Args:
        dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layer in the feedforward.
        dropout (float): The dropout rate.
        heads (int): The number of heads in the multihead attention.

    Attributes:
        dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layer in the feedforward.
        dropout (float): The dropout rate.
        heads (int): The number of heads in the multihead attention.
        norm (nn.LayerNorm): The layer norm.
        q_proj (nn.Linear): The projection of the query.
        k_proj (nn.Linear): The projection of the key.
        v_proj (nn.Linear): The projection of the value.
        attention (MultiheadAttention): The multihead attention.
        feedforward (SimpleFeedForward): The feedforward.

    Input shape: (B, SEQ_LEN, DIM) or regular text shape

    Output shape: (B, SEQ_LEN, DIM) or regular text shape

    Example:
        >>> visual_expert = VisualExpert(1024, 2048, 0.1, 16)
        >>> x = torch.randn(1, 10, 1024)
        >>> out = visual_expert(x)
        >>> out.shape
        torch.Size([1, 10, 1024])

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        heads: int,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.heads = heads

        # Normalization
        self.norm = nn.LayerNorm(dim)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Attention
        self.attention = MultiheadAttention(dim, heads, dropout)

        # Feedforward
        self.feedforward = SimpleFeedForward(dim, hidden_dim, dropout)

    def __call__(self, x: torch.Tensor):
        """Forward pass as shown in the diagram"""

        # Apply Layernorm first
        normalized = self.norm(x)

        # Split into text and image features
        x_text = normalized
        x_image = normalized

        # Apply QKV projections for text
        q_text, k_text, v_text = (
            self.q_proj(x_text),
            self.k_proj(x_text),
            self.v_proj(x_text),
        )

        # Apply QKV projections for image
        q_img, k_img, v_img = (
            self.q_proj(x_image),
            self.k_proj(x_image),
            self.v_proj(x_image),
        )

        # Apply attention where the image features are appended infront of the text features,
        # Concat the q, k, v of text and images together
        q = torch.cat((q_text, q_img))  # , dim=-1)
        k = torch.cat((k_text, k_img))  # , dim=-1)
        v = torch.cat((v_text, v_img))  # , dim=-1)

        # Apply attention
        out = self.attention(q, k, v)

        # Add the output of the attention with the normed x
        out = out + x

        # Another Norm
        normalized = self.norm(out)

        # Seperate text and image features
        out_text = normalized
        out_image = normalized  # torch.split(normalized, self.dim)  # dim=-1)

        # Apply feedforward to both text and image features
        out_text = self.feedforward(out_text)
        out_img = self.feedforward(out_image)

        # Add the output of the feedforwards together with the output of the added attention + norm
        out = out_text + out_img + out

        return out
