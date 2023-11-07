"""
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
from zeta.nn.modules.simple_feedforward import SimpleFeedForward
from zeta.nn.attention.multihead_attention import MultiheadAttention


class VisualExpert:
    def __init_(
        self,
        dim: int,
        hidden_dim: int,
        dropout: int,
        heads: int,
    ):
        self.dim = dim

        # Normalization
        self.norm = nn.LayerNorm(dim)

        # Projections
        self.q_proj = nn.Linear(dim, dim * 3)
        self.k_proj = nn.Linear(dim, dim * 3)
        self.v_proj = nn.Linear(dim, dim * 3)

        # Attention
        self.attention = MultiheadAttention(dim, heads, dropout)

        # Feedforward
        self.feedforward = SimpleFeedForward(dim, hidden_dim, dropout)

    def __call__(self, x):
        # Apply Layernorm first
        x, normalized = self.norm(x)

        # Split into text and image features
        x_text, x_image = torch.split(x, self.dim, dim=-1)

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
        q = torch.cat((q_text, q_img), dim=-1)
        k = torch.cat((k_text, k_img), dim=-1)
        v = torch.cat((v_text, v_img), dim=-1)

        # Apply attention
        out = self.attention(q, k, v)

        # Add the output of the attention with the normed x
        out = out + normalized

        # Another Norm
        normalized = self.norm(out)

        # Seperate text and image features
        out_text, out_image = torch.split(normalized, self.dim, dim=-1)

        # Apply feedforward to both text and image features
        out_text = self.feedforward(out_text)
        out_img = self.feedforward(out_image)

        # Add the output of the feedforwards together with the output of the added attention + norm
        out = out_text + out_img + out

        return out


# x = torch.randn(1, 3, 4, 4)
# ve = VisualExpert(
#     dim=3,
#     hidden_dim=3,
#     dropout=0.1,
#     heads=3,
# )
# out = ve(x)
# print(out.shape)
