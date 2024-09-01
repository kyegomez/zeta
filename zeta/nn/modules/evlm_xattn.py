from zeta.nn.attention.cross_attention import CrossAttention
from torch import nn, Tensor
from zeta.nn.modules.feedforward import FeedForward
from zeta.nn.modules.sparse_moe import NormalSparseMoE


class GatedXAttention(nn.Module):
    """
    GatedXAttention module applies cross attention between text and image embeddings,
    followed by activation functions and feed-forward neural network (FFN) layers.

    Args:
        dim (int): The input dimension of the text embeddings.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.cross_attention = CrossAttention(
            dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            *args,
            **kwargs,
        )

        # ACT
        self.act = nn.Tanh()

        # FFN
        self.ffn = FeedForward(
            dim,
            dim,
            swish=True,
        )

    def forward(self, text: Tensor, img: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass of the GatedXAttention module.

        Args:
            text (Tensor): The input text embeddings. Shape: (batch_size, sequence_length, dim).
            img (Tensor): The input image embeddings.
            mask (Tensor, optional): The attention mask. Defaults to None.

        Returns:
            Tensor: The output tensor after applying cross attention, activation functions, and FFN layers.
        """
        # KV are image, Q is text
        b, s, d = text.shape
        residual = text

        # Cross Attention
        x = self.cross_attention(text, img, mask)

        # Tanh
        feeded = self.act(x)

        # 2nd loop
        out = feeded + residual

        # Second residual
        second_residual = out

        # FFN
        ffn_response = self.ffn(out)

        # Tanded
        out = self.act(ffn_response) + second_residual

        return out


# x = torch.randn(1, 10, 512)
# img = torch.randn(1, 10, 512)

# model = GatedXAttention(512)

# out = model(x, img)
# print(out)


class GatedMoECrossAttn(nn.Module):
    """
    GatedMoECrossAttn is a module that performs gated multi-expert cross attention on text and image inputs.

    Args:
        dim (int): The input dimension.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        experts (int, optional): The number of experts for the MoE. Defaults to 4.

    Attributes:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        cross_attention (CrossAttention): The cross attention module.
        moe (NormalSparseMoE): The MoE module.
        act (Tanh): The activation function.

    Methods:
        forward(text, img, mask=None): Performs forward pass of the module.

    Returns:
        Tensor: The output tensor after the forward pass.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        experts: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        self.cross_attention = CrossAttention(
            dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            *args,
            **kwargs,
        )

        # MoE
        self.moe = NormalSparseMoE(
            dim,
            experts,
        )

        self.act = nn.Tanh()

    def forward(self, text: Tensor, img: Tensor, mask: Tensor = None) -> Tensor:
        residual = text

        # Cross Attention
        attended = self.cross_attention(text, img, mask)

        # Tanh
        activated = self.act(attended) + residual

        # Second Residual
        second_residual = activated

        # MoE
        moe_response, loss = self.moe(activated)

        # Add residual
        out = moe_response + second_residual

        return self.act(out)


# x = torch.randn(1, 10, 512)
# img = torch.randn(1, 10, 512)

# model = GatedMoECrossAttn(512)

# out = model(x, img)
# print(out.shape)
