import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def vit_output_head(x: Tensor, dim: int, num_classes: int = None):
    """
    Applies a Vision Transformer (ViT) output head to the input tensor.

    Args:
        x (Tensor): The input tensor.
        dim (int): The dimension of the input tensor.
        num_classes (int, optional): The number of output classes. Defaults to None.

    Returns:
        Tensor: The output tensor after applying the ViT output head.
    """
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))(x)


def patch_linear_flatten(
    x: Tensor,
    patch_size: int,
    dim: int,
    image_size: int,
    channels: int = 3,
    add_pos_embeddings: bool = False,
    *args,
    **kwargs,
):
    """
    Applies patch embedding to the input tensor and flattens it.

    Args:
        x (Tensor): Input tensor of shape (batch_size, channels, image_height, image_width).
        patch_size (int): Size of the square patch.
        dim (int): Dimension of the output tensor.
        image_size (int): Size of the input image (assumed to be square).
        channels (int, optional): Number of input channels. Defaults to 3.
        add_pos_embeddings (bool, optional): Whether to add positional embeddings. Defaults to False.

    Returns:
        Tensor: Flattened tensor of shape (batch_size, num_patches, dim).
    """
    image_height, image_width = image_size, image_size
    patch_height, patch_width = patch_size, patch_size

    # calculate number of patches
    (image_height // patch_height) * (image_width // patch_width)
    patch_dim = channels * patch_height * patch_width

    # Patch Embedding layer
    to_patch_embeddings = nn.Sequential(
        Rearrange(
            "b c (h p1) (w p2) -> b h w (p1 p2 c)",
            p1=patch_height,
            p2=patch_width,
        ),
        nn.LayerNorm(patch_dim),
        nn.Linear(patch_dim, dim),
        nn.LayerNorm(dim),
    )(x)

    if add_pos_embeddings is not False:
        pos_embeddings = posemb_sincos_2d(x, *args, **kwargs)
        to_patch_embeddings + +pos_embeddings

    return to_patch_embeddings
