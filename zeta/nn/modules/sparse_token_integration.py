"""
Todo:

- Learn more about the taking the images -> converting into patches -> tokens
- Learn more about STI
- Fix current Implementations
- Implement dense channel integration


"""

import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange


# Tokens
# image -> convolution -> tokens -> down sample -> projector
# Image -> average pooling -> concat -> mlp


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


class SparseTokenIntegration(nn.Module):
    """
    SparseTokenIntegration module for integrating sparse tokens into image data.

    Args:
        dim (int): Dimension of the input and output feature vectors.
        num_tokens (int): Number of tokens to be generated.
        image_size (int): Size of the input image (assumed to be square).
        llm_dimension (int): Dimension of the latent linear model.
        channel (int): Number of channels in the input image.
        patch_size (int): Size of the image patch.

    Attributes:
        dim (int): Dimension of the input and output feature vectors.
        num_tokens (int): Number of tokens to be generated.
        image_size (int): Size of the input image (assumed to be square).
        llm_dimension (int): Dimension of the latent linear model.
        channel (int): Number of channels in the input image.
        patch_size (int): Size of the image patch.
        projector (nn.Sequential): Sequential module for projecting the input feature vectors to tokens.
        to_patch_embedding (nn.Sequential): Sequential module for converting image patches to feature vectors.

    """

    def __init__(
        self,
        dim: int = None,
        num_tokens: int = None,
        image_size: int = None,
        llm_dimension: int = None,
        channel: int = 3,
        patch_size: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.image_size = image_size
        self.llm_dimension = llm_dimension
        self.channel = channel
        self.patch_size = patch_size

        # Convolution

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, num_tokens),
        )

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channel * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SparseTokenIntegration module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_tokens).

        """
        b, c, h, w = x.shape
        tokens = self.to_patch_embedding(x)
        print(f"Tokens: {tokens.shape}")

        # Split up for the pathways
        q = tokens
        k = tokens

        # Average pooling
        q = nn.AdaptiveAvgPool1d(self.dim)(q)
        k = nn.AdaptiveAvgPool1d(self.dim)(k)

        print(f"Average Pooling: {q.shape}")
        print(f"Average Pooling: {k.shape}")

        # Concat
        tokens = torch.cat([q, k, tokens], dim=1)
        print(f"Concat: {tokens.shape}")

        return self.projector(tokens)


# x = torch.randn(1, 3, 224, 224)

# model = SparseTokenIntegration(dim=256, num_tokens=512, image_size=224)
# print(model(x).shape)


class SparseChannelIntegration(nn.Module):
    """
    SparseChannelIntegration module integrates sparse tokens into the input image using channel-wise operations.

    Args:
        dim (int): The dimension of the input and output tensors.
        num_tokens (int): The number of tokens to be generated.
        image_size (int): The size of the input image (assumed to be square).
        llm_dimension (int): The dimension of the latent linear model.
        channel (int): The number of channels in the input image.
        patch_size (int): The size of the patches to be extracted from the input image.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        num_tokens (int): The number of tokens to be generated.
        image_size (int): The size of the input image (assumed to be square).
        llm_dimension (int): The dimension of the latent linear model.
        channel (int): The number of channels in the input image.
        patch_size (int): The size of the patches to be extracted from the input image.
        projector (nn.Sequential): The projector network for mapping the input tokens to the output tokens.
        to_patch_embedding (nn.Sequential): The patch embedding network for converting image patches to tokens.

    """

    def __init__(
        self,
        dim: int = None,
        num_tokens: int = None,
        image_size: int = None,
        llm_dimension: int = None,
        channel: int = 3,
        patch_size: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.image_size = image_size
        self.llm_dimension = llm_dimension
        self.channel = channel
        self.patch_size = patch_size

        # Convolution

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, num_tokens),
        )

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channel * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SparseChannelIntegration module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channel, height, width).

        Returns:
            Tensor: The output tensor of shape (batch_size, num_tokens).

        """
        b, c, h, w = x.shape
        tokens = self.to_patch_embedding(x)
        print(f"Tokens: {tokens.shape}")

        # Split up for the pathways
        q = tokens
        k = tokens

        # Concat
        tokens = torch.cat([q, k, tokens], dim=1)
        print(f"Concat: {tokens.shape}")

        return self.projector(tokens)


# x = torch.randn(1, 3, 224, 224)

# model = SparseChannelIntegration(dim=256, num_tokens=512, image_size=224)

# print(model(x))
