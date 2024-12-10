import torch
from torch import nn


class VisionEmbedding(nn.Module):
    """
    Image to Patch Embedding


    Args:
        img_size (int): The image size.
        patch_size (int): The patch size.
        in_chans (int): The number of input channels.
        embed_dim (int): The embedding dimension.
        contain_mask_token (bool): Whether to contain mask token or not.
        prepend_cls_token (bool): Whether to prepend cls token or not.

    Attributes:
        patch_shape (tuple): The patch shape.
        img_size (tuple): The image size.
        patch_size (tuple): The patch size.
        num_patches (int): The number of patches.
        proj (nn.Module): The projection layer.
        mask_token (nn.Parameter): The mask token.
        cls_token (nn.Parameter): The cls token.

    Example:
        >>> module = VisionEmbedding(224, 16, 3, 768)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 197, 768])

    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        contain_mask_token=False,
        prepend_cls_token=False,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]
        )
        self.patch_shape = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if contain_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        """num_position_embeddings"""
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        """forward"""
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model"
            f" ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        return x
