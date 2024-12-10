import torch
from torch import nn


class VisionLanguageEmbedding(nn.Module):
    """
    Vision Language Embedding.

    Args:
        text_embed (nn.Module): The text embedding layer.
        vision_embed (nn.Module): The vision embedding layer.

    Example:
        >>> module = VisionLanguageEmbedding(nn.Embedding(10, 10), nn.Embedding(10, 10))
        >>> x = torch.randn(10, 10)
        >>> y = module(x, x)
        >>> y.shape
        torch.Size([10, 20])

    """

    def __init__(self, text_embed, vision_embed):
        super().__init__()
        self.text_embed = text_embed
        self.vision_embed = vision_embed

    def forward(self, textual_tokens, visual_tokens, **kwargs):
        if textual_tokens is None:
            return self.vision_embed(visual_tokens)

        if visual_tokens is None:
            return self.text_embed(textual_tokens)

        x1 = self.vision_embed(visual_tokens)
        x2 = self.text_embed(textual_tokens)

        return torch.cat([x1, x2], dim=1)
