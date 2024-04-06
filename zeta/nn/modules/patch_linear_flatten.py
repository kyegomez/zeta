import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from einops import repeat


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


def vit_output_head(
    x: Tensor, dim: int, num_classes: int = None, pooling: str = "mean"
):
    """
    Applies a Vision Transformer (ViT) output head to the input tensor.

    Args:
        x (Tensor): The input tensor.
        dim (int): The dimension of the input tensor.
        num_classes (int, optional): The number of output classes. Defaults to None.

    Returns:
        Tensor: The output tensor after applying the ViT output head.
    """
    if pooling == "mean":
        x = x.mean(dim=1)
    elif pooling == "cls":
        x = x[:, 0]
    elif pooling == "max":
        x = x.max(dim=1).values
    elif pooling == "none":
        x = x
    x = nn.Identity()(x)  # Identity layer to avoid error in nn.Sequential
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


def video_patch_linear_flatten(
    x: Tensor,
    patch_size: int,
    dim: int,
    image_size: int,
    channels: int = 3,
    add_pos_embeddings: bool = False,
    frame_patch_size: int = 1,
    frames: int = None,
    seqlen: int = None,
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

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by frame patch size"

    # calculate number of patches
    num_patches = (
        (image_height // patch_height)
        * (image_width // patch_width)
        * (frames // frame_patch_size)
    )
    patch_dim = channels * patch_height * patch_width * frame_patch_size

    # Patch Embedding layer
    to_patch_embeddings = nn.Sequential(
        Rearrange(
            "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
            p1=patch_height,
            p2=patch_width,
            pf=frame_patch_size,
        ),
        nn.LayerNorm(patch_dim),
        nn.Linear(patch_dim, dim),
        nn.LayerNorm(dim),
    )(x)

    if add_pos_embeddings is not False:
        pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        to_patch_embeddings += pos_embedding[:, : (seqlen + 1)]

    return to_patch_embeddings


def cls_tokens(
    x: Tensor,
    dropout: float = 0.0,
    num_patches: int = None,
    pos_emb: bool = False,
):
    """
    Adds class tokens to the input tensor and applies dropout and positional embeddings if specified.

    Args:
        x (Tensor): The input tensor of shape (batch_size, sequence_length, hidden_dim).
        dropout (float, optional): The dropout probability. Defaults to 0.0.
        num_patches (int, optional): The number of patches. Defaults to None.
        pos_emb (bool, optional): Whether to apply positional embeddings. Defaults to False.

    Returns:
        Tensor: The modified input tensor with class tokens added.

    """
    b, s, d = x.shape

    cls_tokens = repeat(x, "1 1 d -> b 1 d", b=b)
    x = torch.cat((cls_tokens, x), dim=1)

    if dropout is not None:
        x = nn.Dropout(dropout)(x)

    if pos_emb:
        pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, d))
        x += pos_embeddings[:, : (s + 1)]

    return x


# # video: b, c, f, h, w
# x = torch.randn(1, 3, 16, 224, 224)

# # patch size
# patch_size = 16
# frames = 16
# frame_patch_size = 1
# dim = 512
# image_size = 224
# channels = 3
# model = video_patch_linear_flatten(
#     x, patch_size, dim, image_size, channels, frames=frames, frame_patch_size=frame_patch_size
# )

# print(model.shape)
