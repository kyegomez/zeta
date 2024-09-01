from typing import List

import torch
from torch import Tensor, nn


class MultiModalEmbedding(nn.Module):
    """
    MultiModalEmbedding class represents a module for multi-modal embedding.

    Args:
        video_dim (int): The dimension of the video input.
        text_dim (int): The dimension of the text input.

    Attributes:
        video_embedding (nn.Linear): Linear layer for video embedding.
        text_embedding (nn.EmbeddingBag): Embedding layer for text embedding.

    Methods:
        forward(video, text): Performs forward pass of the multi-modal embedding.

    Returns:
        torch.Tensor: Concatenated tensor of video and text embeddings.
    """

    def __init__(self, video_dim, text_dim):
        super().__init__()
        self.video_embedding = nn.Linear(video_dim, 512)
        self.text_embedding = nn.EmbeddingBag(text_dim, 512, sparse=True)

    def forward(self, video, text):
        video_embed = self.video_embedding(video)
        text_embed = self.text_embedding(text)
        return torch.cat([video_embed, text_embed], dim=-1)


class MultiInputMultiModalConcatenation(nn.Module):
    """
    A module that concatenates multiple input tensors along a specified dimension.

    Args:
        dim (int): The dimension along which the input tensors will be concatenated.

    Attributes:
        dim (int): The dimension along which the input tensors will be concatenated.
    """

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: List[Tensor]):
        """
        Forward pass of the module.

        Args:
            inputs (List[Tensor]): A list of input tensors to be concatenated.

        Returns:
            Tensor: The concatenated tensor.
        """
        return torch.cat(inputs, dim=self.dim)


class SplitMultiOutput(nn.Module):
    """
    Splits the input tensor into multiple outputs along a specified dimension.

    Args:
        dim (int): The dimension along which to split the input tensor.
        num_splits (int): The number of splits to create.
        output_dims (List[int]): The sizes of the output tensors along the split dimension.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The dimension along which to split the input tensor.
        num_splits (int): The number of splits to create.
        output_dims (List[int]): The sizes of the output tensors along the split dimension.
    """

    def __init__(
        self,
        dim: int,
        num_splits: int,
        output_dims: List[int],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_splits = num_splits
        self.output_dims = output_dims

    def forward(self, x: Tensor):
        """
        Forward pass of the SplitMultiOutput module.

        Args:
            x (Tensor): The input tensor to be split.

        Returns:
            Tuple[Tensor]: A tuple of output tensors after splitting the input tensor.
        """
        return torch.split(x, self.output_dims, dim=self.dim)


class OutputHead(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_range: int = 1,
        vocab_size: int = 20000,
        *args,
        **kwargs,
    ):
        """
        Initializes an OutputHead module.

        Args:
            dim (int): The input dimension.
            dim_range (int): The dimension range for softmax operation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.dim_range = dim_range

        # Linear layer for each output
        self.output_layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
            nn.Softmax(dim_range),
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the OutputHead module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.output_layers(x)


class DynamicOutputDecoder(nn.Module):
    """
    Decoder module for dynamic output.

    Args:
        dim (int): The input dimension.
        robot_count (int): The number of robots.

    Attributes:
        decoders (nn.ModuleList): List of linear decoders.

    """

    def __init__(self, dim, robot_count):
        super().__init__()
        self.decoders = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(robot_count)]
        )

    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            List[torch.Tensor]: List of decoded tensors.

        """
        return [decoder(x) for decoder in self.decoders]


class DynamicInputChannels(nn.Module):
    """
    A module that applies linear transformations to input data for multiple robots.

    Args:
        num_robots (int): The number of robots.
        dim (int): The input dimension.
        output_dim (int): The output dimension.

    Attributes:
        layers (nn.ModuleList): A list of linear layers.

    Methods:
        forward(x): Forward pass of the module.

    """

    def __init__(self, num_robots, dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, output_dim) for _ in range(num_robots)]
        )

    def forward(self, x):
        outputs = [layer(x) for layer in self.layers]
        return torch.cat(outputs, dim=1)


class OutputDecoders(nn.Module):
    """
    Class representing the output decoders for multiple robots.

    Args:
        num_robots (int): The number of robots.
        dim (int): The input dimension.
        output_dim (int): The output dimension.

    Attributes:
        decoders (nn.ModuleList): List of linear decoders for each robot.

    Methods:
        forward(x): Forward pass of the decoders.

    """

    def __init__(self, num_robots, dim, output_dim):
        super().__init__()
        self.decoders = nn.ModuleList(
            [nn.Linear(dim, output_dim) for _ in range(num_robots)]
        )

    def forward(self, x):
        """
        Forward pass of the decoders.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Stacked output tensor from each decoder.

        """
        return torch.stack([decoder(x) for decoder in self.decoders], dim=1)
