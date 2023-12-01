import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MLPBlock(nn.Module):
    """MLPBlock

    Args:
        dim (int): [description]
    """

    def __init__(self, dim: int):
        super(MLPBlock, self).__init__()
        self.dense1 = nn.Linear(dim, dim)
        self.dense2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLPBlock

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        y = self.dense1(x)
        y = F.gelu(y)
        return self.dense(y)


class MixerBlock(nn.Module):
    """MixerBlock


    Args:
        mlp_dim (int): [description]
        channels_dim (int): [description]
    """

    def __init__(self, mlp_dim: int, channels_dim: int):
        super(MixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels_dim)
        self.tokens_mlp = MLPBlock(mlp_dim)

        self.norm2 = nn.LayerNorm(channels_dim)
        self.channel_mlp = MLPBlock(mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MixerBlock

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        y = self.norm1(x)
        y = rearrange(y, "n c t -> n t c")
        y = self.tokens_mlp(y)
        y = rearrange(y, "n t c -> n c t")
        x = x + y
        y = self.norm2(x)
        return x + self.channel_mlp(y)


class MLPMixer(nn.Module):
    """MLPMixer

    Args:
        num_classes (int): [description]
        num_blocks (int): [description]
        patch_size (int): [description]
        hidden_dim (int): [description]
        tokens_mlp_dim (int): [description]
        channels_mlp_dim (int): [description]

    Examples:
        >>> from zeta.nn import MLPMixer
        >>> model = MLPMixer(10, 8, 16, 32, 64, 64)
        >>> x = torch.randn(32, 3, 224, 224)
        >>> model(x).shape
        torch.Size([32, 10])


    """

    def __init__(
        self,
        num_classes: int,
        num_blocks: int,
        patch_size: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ):
        super(MLPMixer, self).__init__()
        self.stem = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(tokens_mlp_dim, channels_mlp_dim)
                for _ in range(num_blocks)
            ]
        )
        self.pred_head_layernorm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLPMixer

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.stem(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.pred_head_layernorm(x)
        x = x.mean(dim=1)
        return self.head(x)


# Example of creating a model instance
mlp_mixer = MLPMixer(
    num_classes=10,
    num_blocks=8,
    patch_size=16,
    hidden_dim=512,
    tokens_mlp_dim=256,
    channels_mlp_dim=512,
)

# Example input tensor
example_input = torch.randn(
    1, 512, 32, 32
)  # Batch size of 1, 512 channels, 32x32 image
output = mlp_mixer(example_input)
print(
    output.shape
)  # Should output the shape corresponding to the number of classes
