from torch import nn


class ImgPatchEmbed(nn.Module):
    """patch embedding module


    Args:
        img_size (int, optional): image size. Defaults to 224.
        patch_size (int, optional): patch size. Defaults to 16.
        in_chans (int, optional): input channels. Defaults to 3.
        embed_dim (int, optional): embedding dimension. Defaults to 768.

    Examples:
        >>> x = torch.randn(1, 3, 224, 224)
        >>> model = ImgPatchEmbed()
        >>> model(x).shape
        torch.Size([1, 196, 768])


    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """Forward

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
