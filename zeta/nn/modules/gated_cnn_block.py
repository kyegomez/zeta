import torch
from torch import nn, Tensor


# [MAIN]
class GatedCNNBlock(nn.Module):
    def __init__(
        self,
        dim: int = None,
        expansion_ratio: float = 8 / 3,
        kernel_size: int = 7,
        conv_ratio: float = 1.0,
        drop_path: float = 0.0,
        *args,
        **kwargs,
    ):
        super(GatedCNNBlock, self).__init__()
        self.dim = dim
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size
        self.conv_ratio = conv_ratio
        self.drop_path = drop_path
        self.hidden = int(expansion_ratio * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        self.g_act = nn.GroupNorm(1, dim)

        # Linear layers
        self.fc1 = nn.Linear(dim, self.hidden * 2)
        self.fc2 = nn.Linear(self.hidden, dim)

        # Conv chanels
        self.conv_channels = int(conv_ratio * dim)
        self.split_indices = (
            self.hidden,
            self.hidden - self.conv_channels,
            self.conv_channels,
        )
        self.conv = nn.Conv2d(
            self.conv_channels,
            self.conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.conv_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        # Normalize
        x = self.norm(x)

        # Torch split
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)

        # C
        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)

        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x + shortcut


# # Forward example
# x = torch.randn(1, 3, 64, 64)

# model = GatedCNNBlock(
#     dim = 3,
# )

# print(model(x).shape)
