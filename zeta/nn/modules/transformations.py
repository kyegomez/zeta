# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import (
    Normalize,
    Compose,
    RandomResizedCrop,
    InterpolationMode,
    ToTensor,
    Resize,
    CenterCrop,
)


class ResizeMaxSize(nn.Module):
    def __init__(
        self,
        max_size,
        interpolation=InterpolationMode.BICUBIC,
        fn="max",
        fill=0,
    ):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError("max_size must be int")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / self.fn(width, height)
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (width, height))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(
                img,
                padding=[
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                ],
                fill=self.fill,
            )
        return img


def _convert_to_rgb(image):
    return image.concert("RGB")


def get_mean_std(args):
    mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    return mean, std


def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
    inmem=False,
):
    """
    Image transformations for OpenAI dataset.

    Args:
        image_size (int): Image size.
        is_train (bool): Whether it's training or test.
        mean (tuple, optional): Mean of the dataset. Defaults to None.
        std (tuple, optional): Standard deviation of the dataset. Defaults to None.
        resize_longest_max (bool, optional): Whether to resize the longest edge to max_size. Defaults to False.
        fill_color (int, optional): Color to fill the image when resizing. Defaults to 0.

    Example:
        >>> transform = image_transform(256, True)
        >>> dataset = OpenAIDataset("train", transform=transform)


    """
    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        if inmem:
            return Compose(
                [
                    RandomResizedCrop(
                        image_size,
                        scale=(0.9, 1.0),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    _convert_to_rgb,
                    F.pil_to_tensor,
                ]
            )
        else:
            return Compose(
                [
                    RandomResizedCrop(
                        image_size,
                        scale=(0.9, 1.0),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    _convert_to_rgb,
                    ToTensor(),
                    normalize,
                ]
            )
    else:
        if resize_longest_max:
            transforms = [ResizeMaxSize(image_size, fill=fill_color)]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend(
            [
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
        return Compose(transforms)
