from torch import nn, Tensor
from typing import Optional
from zeta import FeedForward
from zeta import MultiModalCrossAttention, MultiQueryAttention
from zeta.nn.embeddings.mi_rope import MIRoPE
import torch
from torch.nn import functional as F


def exists(val):
    return val is not None


class AdaptiveGating(nn.Module):
    def __init__(self, hidden_dim: int):
        """
        Initializes an instance of the AdaptiveGating class.

        Args:
            hidden_dim (int): The dimension of the hidden state.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hat_text: Tensor,
        bar_text: Tensor,
    ) -> Tensor:
        """
        Performs the forward pass of the AdaptiveGating module.

        Args:
            hat_text (Tensor): The input tensor representing the hat text.
            bar_text (Tensor): The input tensor representing the bar text.

        Returns:
            Tensor: The fused hidden state tensor.

        """
        g = self.sigmoid(hat_text)

        # Step 2
        h_fused = bar_text * g + hat_text * (1 - g)

        return h_fused


class HyperAttentionmPLUGOwlBlock(nn.Module):
    """
    HyperAttentionmPLUGOwlBlock is a module that performs hyper attention between image and text inputs.

    Args:
        dim (int): The dimension of the input.
        heads (int): The number of attention heads.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        mi_rope_on (bool, optional): Whether to use mutual information rope. Defaults to True.
        max_seq_len (int, optional): The maximum sequence length. Defaults to 100.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        mi_rope_on: bool = True,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mi_rope_on = mi_rope_on
        self.max_seq_len = max_seq_len

        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swiglu=True,
        )

        # Projections
        self.w_kv_img = nn.Linear(dim, dim * 2)
        self.w_q_text = nn.Linear(dim, dim)

        # Attention
        self.attn = MultiModalCrossAttention(
            dim,
            heads,
            context_dim=dim,
            qk=True,
            post_attn_norm=True,
        )

        self.attn_op = MultiQueryAttention(
            dim,
            heads,
        )

        # Rotary Position Embedding
        self.rotary_embeddings = MIRoPE(dim)
        self.proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.final_proj = nn.Linear(dim, dim)
        self.gate = AdaptiveGating(dim)

    def forward(self, img: Tensor, text: Tensor, mask: Optional[Tensor] = None):
        """
        Forward pass of the HyperAttentionmPLUGOwlBlock module.

        Args:
            img (Tensor): The input image tensor.
            text (Tensor): The input text tensor.
            mask (Optional[Tensor], optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        img.shape[1]
        img = self.norm(img)
        text = self.norm(text)

        # Rotary Position Embedding
        # positions, scale = self.get_rotary_embedding(n, img.device)

        # Apply rotary position embedding

        w_img_k = self.proj(img)
        w_img_v = self.proj(img)

        w_q_text = self.text_proj(text)
        w_k_text = self.text_proj(w_q_text)
        w_v_text = self.text_proj(w_q_text)

        # Attn op
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            img_attn = F.scaled_dot_product_attention(
                w_q_text,
                w_img_k,
                w_img_v,
            )

        with torch.backends.cuda.sdp_kernel(enable_math=True):
            text_attn = F.scaled_dot_product_attention(
                w_q_text,
                w_k_text,
                w_v_text,
            )

        output_gate = self.gate(img_attn, text_attn)

        return self.final_proj(output_gate)

    def get_rotary_embedding(self, n, device):
        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]

        pos_emb, scale = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        self.register_buffer("pos_emb_scale", scale, persistent=False)
        return pos_emb, scale


input = torch.randn(1, 10, 512)

conditioning = torch.randn(1, 10, 512)
model = HyperAttentionmPLUGOwlBlock(512, 8)
output = model(input, conditioning)
print(output.shape)
