import torch
from torch import nn, Tensor
from zeta.nn import (
    MultiQueryAttention,
    FeedForward,
    patch_linear_flatten,
    vit_output_head,
)
from einops import reduce


class TransformerBlock(nn.Module):
    """
    TransformerBlock is a module that represents a single block in a transformer network.

    Args:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int, optional): The multiplier for the hidden dimension in the feedforward network. Defaults to 4.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
            # qk_ln=True,
        )

        # Feedforward
        self.ffn = FeedForward(
            dim,
            dim,
            mult,
            swish=True,
            post_act_ln=True,
            dropout=dropout,
        )

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        skip = x

        # Norm
        x = self.norm(x)

        # Attention
        x, _, _ = self.attn(x)
        x + skip

        # Skip2
        skip_two = x

        # Norm
        x = self.norm(x)

        # Feedforward
        return self.ffn(x) + skip_two


class Spectra(nn.Module):
    """
    Spectra class represents a neural network model for image classification using the Vision Transformer (ViT) architecture.

    Args:
        dim (int): The dimension of the model.
        heads (int): The number of attention heads in the model.
        dim_head (int): The dimension of each attention head.
        mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        patch_size (int, optional): The size of each patch in the image. Defaults to 16.
        image_size (int, optional): The size of the input image. Defaults to 224.
        num_classes (int, optional): The number of output classes. Defaults to 1000.
        depth (int, optional): The number of transformer blocks in the model. Defaults to 8.
        channels (int, optional): The number of input channels in the image. Defaults to 3.

    Attributes:
        dim (int): The dimension of the model.
        heads (int): The number of attention heads in the model.
        dim_head (int): The dimension of each attention head.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        dropout (float): The dropout rate.
        patch_size (int): The size of each patch in the image.
        image_size (int): The size of the input image.
        num_classes (int): The number of output classes.
        depth (int): The number of transformer blocks in the model.
        channels (int): The number of input channels in the image.
        layers (nn.ModuleList): The list of transformer blocks in the model.
        norm (nn.LayerNorm): The layer normalization module.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.0,
        patch_size: int = 16,
        image_size: int = 224,
        num_classes: int = 1000,
        depth: int = 8,
        channels: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.depth = depth
        self.channels = channels

        # Layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, mult, dropout)
                for _ in range(depth)
            ]
        )

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the Spectra model.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: The output tensor of shape (batch_size, num_classes).
        """
        # Patch Image
        x = patch_linear_flatten(
            x,
            self.patch_size,
            self.dim,
            self.image_size,
            self.channels,
        )
        print(f"Patch Image Shape: {x.shape}")
        x = reduce(x, "b h w c -> b (h w) c", "mean")
        print(x.shape)

        # Apply layers
        for layer in self.layers:
            x = layer(x)

        # Norm
        x = self.norm(x)

        # VIT output head
        out = vit_output_head(x, self.dim, self.num_classes)
        return out


# Img shape [B, C, H, W]
img = torch.randn(1, 3, 224, 224)


# Model
# Img -> patch -> linear -> flatten -> transformer layers -> output classification
model = Spectra(
    dim=512,
    heads=8,
    dim_head=64,
    mult=4,
    dropout=0.0,
    patch_size=16,
    image_size=224,
    num_classes=1000,
    depth=8,
    channels=3,
)

# Forward
out = model(img)
print(out)
print(out.shape)
