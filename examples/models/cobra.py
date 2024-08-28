import torch
from torch import nn, Tensor
from zeta import SSM

# from zeta.nn.modules import TextTokenEmbedding


class CobraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        channels: int = 64,
    ):
        super().__init__()

        # Projection
        self.proj = nn.Linear(dim, dim)

        # Convolution -- output the same shap
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=1,
        )

        # Activation
        self.swish = nn.SiLU()

        # Init SSM
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

    def forward(self, x: Tensor):
        # Create 2 pathways
        skip = x

        # Split up the paths
        x_one = self.proj(x)
        x_two = self.proj(x)
        print(x_two.shape)
        print(x_one.shape)

        # Apply the convolution
        x_one = self.conv(x_one)
        print(x_one.shape)

        # Apply the activation
        x_one = self.swish(x_one)

        # Apply the SSM
        x_one = self.ssm(x_one)
        print(x_one.shape)

        # Apply the activation
        x_two = self.swish(x_two)

        # Matmul
        out = x_one * x_two

        # Add the skip connection
        out = out + skip

        return self.proj(out)


# x = torch.randn(1, 64, 256)

# block = CobraBlock(
#     dim = 256,
#     dt_rank = 8,
#     dim_inner = 256,
#     d_state = 256
# )

# out = block(x)
# print(out)


class Cobra(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        channels: int = 64,
        num_tokens: int = 10000,
        depth: int = 12,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.channels = channels
        self.num_tokens = num_tokens
        self.depth = depth

        # Token Embedding
        # self.embed = TextTokenEmbedding(
        #     dim,
        #     num_tokens,
        #     l2norm_embed=True
        # )
        self.embed = nn.Embedding(num_tokens, dim)

        # Layers
        self.layers = nn.ModuleList(
            [
                CobraBlock(
                    dim, dt_rank, dim_inner, d_state, channels, *args, **kwargs
                )
                for _ in range(depth)
            ]
        )

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Embed
        x = self.embed(x)
        x = self.norm(x)

        # Loop through the layers
        for layer in self.layers:
            x = layer(x)

        # Norm
        x = self.norm(x)
        return x


# Forward pass
x = torch.randint(0, 10000, (1, 64))

model = Cobra(
    dim=256,
    dt_rank=8,
    dim_inner=256,
    d_state=256,
    channels=64,
    num_tokens=10000,
    depth=12,
)

out = model(x)
print(out)
