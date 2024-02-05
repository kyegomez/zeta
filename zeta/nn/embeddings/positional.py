import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class PositionalEmbedding(nn.Embedding):
    """PositionalEmbedding module.


    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the input sequence.
        padding_idx (int, optional): Index of the padding token. Defaults to 0.
        scale_grad_by_freq (bool, optional): If True, scale gradients by frequency. Defaults to False.
        sparse (bool, optional): If True, use sparse gradient updates. Defaults to False.

    Example:
        >>> positional_embedding = PositionalEmbedding(512, 1000)
        >>> x = torch.randn(32, 100, 512)
        >>> positions = torch.arange(100)
        >>> embedded_tensor = positional_embedding(x, positions)
    """

    def forward(
        self,
        x,
        positions=None,
        **kwargs,
    ):
        """
        Forward pass of the PositionalEmbedding module.

        Args:
            x (torch.Tensor): Input tensor.
            positions (torch.Tensor, optional): Positions tensor. If None, positions are generated based on the input tensor size. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Embedded tensor.

        """
        if positions is None:
            # being consistent with Fairseq, which starts from 2.
            positions = (
                torch.arange(2, x.size(1) + 2, device=x.device)
                .long()
                .unsqueeze(0)
            )

        positions = rearrange(positions, "b l -> l b")
        x = rearrange(x, "b l d -> l b d")
        embedded_tensor = F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        embedded_tensor = rearrange(embedded_tensor, "l b d -> b l d")

        return embedded_tensor
