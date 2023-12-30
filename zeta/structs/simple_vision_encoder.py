import torch
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
from typing import Tuple
from torch import nn
from huggingface_hub import snapshot_download


class VisionEncoder(nn.Module):
    """
    Initializes a VisionEncoder object.

    Args:
        size (Tuple, optional): The size of the input image. Defaults to (384, 384).
        model_path (str, optional): The path to the pre-trained vision model. Defaults to "model".
        return_shape (bool, optional): Whether to return the shape of the embedding. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples::
        >>> from zeta.structs import VisionEncoder
        >>> encoder = VisionEncoder()
        >>> embeds = encoder("image.jpg")
        >>> embeds.shape
        torch.Size([1, 512])
    """

    def __init__(
        self,
        size: Tuple = (384, 384),
        model_name: str = "vikhyatk/moondream0",
        return_shape: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.size = size
        self.model_name = model_name
        self.return_shape = return_shape
        model_path = snapshot_download(model_name)

        self.model = torch.jit.load(f"{model_path}/vision.pt").to(
            dtype=torch.float32
        )

        self.preprocess = Compose(
            [
                Resize(size=size, interpolation=InterpolationMode.BICUBIC),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                *args,
            ]
        )

    def __call__(self, image: Image, *args, **kwargs) -> torch.Tensor:
        """
        Processes an input image and returns its embedding.

        Args:
            image (Image): The input image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The embedding of the input image.
        """
        image = Image.open(image)
        with torch.no_grad():
            image_vec = self.preprocess(image.convert("RGB")).unsqueeze(0)
            embeds = self.model(image_vec, *args, **kwargs)

            if self.return_shape:
                print(f"Embedding shape: {embeds.shape}")

            return embeds
