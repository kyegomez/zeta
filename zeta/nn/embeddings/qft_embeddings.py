import numpy as np
import torch
from torch import nn


class QFTSPEmbeddings(nn.Module):
    """Quantum Fourier Transform-inspired Shift Phase Embeddings.


    Attributes:
        vocab_size (int): The size of the vocabulary.
        dim (int): The dimensionality of the embeddings.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass of the QFTSPEmbeddings module.

    Example:
        >>> vocab_size = 10000
        >>> dim = 512
        >>> model = QFTSPEmbeddings(vocab_size, dim)
        >>> x = torch.randint(0, vocab_size, (1, 10))
        >>> embeddings = model(x)
        >>> print(embeddings)
    """

    def __init__(
        self, vocab_size: int = None, dim: int = None, *args, **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embeddings = nn.Embedding(vocab_size, dim, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the QFTSPEmbeddings module.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: phase shifted embeddings
        """
        # real valued embeddings
        embeds = self.embeddings(x)

        # Quantum-inspired operation: Phase shift
        # Split embed_dim into two halves for real and imaginary parts
        phase_shift = torch.exp(2j * np.pi * torch.rand(self.dim // 2))
        shifted_embeds = torch.cat(
            [
                embeds[:, :, : self.dim // 2] * phase_shift.real,
                embeds[:, :, self.dim // 2 :] * phase_shift.imag,
            ],
            dim=-1,
        )

        return shifted_embeds
