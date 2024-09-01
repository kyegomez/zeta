import torch
import torch.nn as nn
from typing import Tuple


class CROMEAdapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        """
        Initialize the CROMEAdapter module.

        Args:
            input_dim (int): The dimension of the input features.
            bottleneck_dim (int): The dimension of the bottleneck layer.
        """
        super(CROMEAdapter, self).__init__()

        self.Wd_text = nn.Linear(input_dim, bottleneck_dim)
        self.Wg_text = nn.Linear(input_dim, bottleneck_dim)
        self.Wd_image = nn.Linear(input_dim, bottleneck_dim)
        self.Wg_image = nn.Linear(input_dim, bottleneck_dim)

        self.Wu = nn.Linear(bottleneck_dim, input_dim)

        self.silu = nn.SiLU()

    def forward(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass of the CROMEAdapter module.

        Args:
            text_features (torch.Tensor): The input text features.
            image_features (torch.Tensor): The input image features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output text and image features.
        """
        text_down = self.silu(self.Wd_text(text_features)) * self.Wg_text(
            text_features
        )
        image_down = self.silu(self.Wd_image(image_features)) * self.Wg_image(
            image_features
        )
        text_up = self.Wu(text_down)
        image_up = self.Wu(image_down)
        text_output = text_features + text_up
        image_output = image_features + image_up

        return text_output, image_output


# model = CROMEAdapter(512, 256)
# text_features = torch.randn(1, 2, 512)
# image_features = torch.randn(1, 2, 512)
# output_text, output_image = model(text_features, image_features)
# print(output_text.shape, output_image.shape)
