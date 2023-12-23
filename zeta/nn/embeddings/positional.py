import torch
import torch.nn.functional as F
from torch import nn


class PositionalEmbedding(nn.Embedding):
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

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
