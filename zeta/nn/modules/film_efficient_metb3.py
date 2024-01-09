from torch import nn, Tensor
from zeta.nn.modules.mbconv import MBConv
from zeta.nn.modules.film import Film


class FiLMEfficientNetB3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        downsample: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout: float = 0.1,
        num_mbconv_blocks: int = 26,
        num_film_layers: int = 26,
        expanse_ratio: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_mbconv_blocks = num_mbconv_blocks
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_film_layers = num_film_layers
        self.expanse_ratio = expanse_ratio
        self.hidden_dim = dim * expanse_ratio

        for _ in range(num_mbconv_blocks):
            self.mb_conv_layers = nn.ModuleList(
                [
                    MBConv(
                        dim,
                        dim,
                        downsample=downsample,
                        dropout=dropout,
                        *args,
                        **kwargs,
                    )
                ]
            )

            self.film_layers = nn.ModuleList(
                [Film(dim, self.hidden_dim, expanse_ratio=expanse_ratio)]
            )

        self.proj = nn.Linear(in_channels, out_channels)

    def forward(
        self, text: Tensor, img: Tensor, weight: Tensor = None, *args, **kwargs
    ) -> Tensor:
        x = img

        # Apply MBConv and film layers
        for mb_conv, film in zip(self.mbconv_layers, self.film_layers):
            x = mb_conv(x)
            x = film(x, text)

        # Flatten the output to pass through the projection layer
        x = x.view(x.size(0), -1)
        x = self.proj(x)

        return x


# x = torch.randn(1, 3, 224, 224)
# text = torch.randn(1, 128)
# model = FiLMEfficientNetB3(
#     in_channels=3,
#     out_channels=1000,
#     dim=128,
#     downsample=1,
#     kernel_size=3,
#     stride=1,
#     padding=1,
#     dropout=0.1,
#     num_mbconv_blocks=26,
#     num_film_layers=26,
#     expanse_ratio=4,
# )
# output = model(text, x)
# print(output.shape)
