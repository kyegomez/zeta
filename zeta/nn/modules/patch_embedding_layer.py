from torch import nn, Tensor
from zeta.nn.modules.patch_img import patch_img
from zeta.nn.attention.cross_attention import CrossAttention

# from zeta.nn.modules.feedforward import Feedforward


class PatchEmbeddingLayer(nn.Module):
    def __init__(
        self,
        dim: int = None,
        patches: int = 16,
        image_size: int = 224,
        in_channels: int = 3,
    ):
        super(PatchEmbeddingLayer, self).__init__()
        self.dim = dim
        self.patches = patches
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_dim = in_channels * patches**2
        self.patch_size = image_size // patches
        self.num_patches = (image_size // self.patch_size) ** 2

        self.cross_attn = CrossAttention(dim=dim, context_dim=self.dim)
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Linear(dim, dim * 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        patches = patch_img(
            x,
            patches=self.patches,
        )
        print(patches.shape)
        b, s, d = patches.shape

        # Run cross attn
        # attended = self.cross_attn(patches, patches)
        attended = CrossAttention(dim=d, context_dim=self.dim)(patches, patches)
        print(attended.shape)

        # Flatten patches
        out = self.ffn(attended)
        print(out.shape)

        return out


# x = torch.randn(1, 3, 224, 224)

# model = PatchEmbeddingLayer(
#     dim = 224,
#     patches = 16,
#     image_size = 224,
#     in_channels = 3
# )

# out = model(x)
# print(out.shape)
