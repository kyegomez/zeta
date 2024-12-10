import torch
from einops import rearrange


def embedding_to_grid(embedding: torch.Tensor, grid_size):
    """
    Embedding to grid

    Einstein summation notation:
        'b' = batch size
        'h' = image height
        'w' = image width
        'd' = embedding dimension

    Args:
        embedding (torch.Tensor): Embedding to be reshaped
        grid_size (tuple): Grid size

    Returns:
        torch.Tensor: Reshaped embedding

    Usage:
        >>> embedding = torch.rand(1, 128, 4, 4)
        >>> embedding.shape
        torch.Size([1, 128, 4, 4])
        >>> embedding = embedding_to_grid(embedding, (8, 8))
        >>> embedding.shape
        torch.Size([1, 128, 8, 8])

    """
    h, w = grid_size
    return rearrange(embedding, "b (h w d) -> b d h w ", h=h, w=w)
