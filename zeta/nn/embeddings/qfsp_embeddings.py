import torch
import torch.nn as nn
import torch.nn.functional as F

# QFTSPEmbedding
class QFTSPEmbedding(nn.Module):
    """
    QFTSPEmbedding with multiple collapse mechanisms.

    This module allows for different ways of collapsing the superposition of embeddings,
    based on the provided context and selected mechanism.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        collapse_mode: str = "weighted_sum",
        **kwargs,
    ):
        super(QFTSPEmbedding, self).__init__()
        self.dim = dim
        self.collapse_mode = collapse_mode
        self.base_embeddings = nn.Embedding(vocab_size, dim)
        self.superposed_embeddings = nn.Embedding(vocab_size, dim)
        self.linear_transform = nn.Linear(2 * dim, dim)

    def forward(
        self, x: torch.Tensor, context_vector: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the QFTSPEmbedding module.

        Args:
            x (_type_): _description_
            context_vector (_type_): _description_
            collapse_mode (str, optional): _description_. Defaults to "weighted_sum".

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        base_embeds = self.base_embeddings(x)
        superposed_embeds = self.superposed_embeddings(x)

        if self.collapse_mode == "weighted_sum":
            collapsed_embeds = (
                base_embeds + context_vector.unsqueeze(-1) * superposed_embeds
            )
        elif self.collapse_mode == "dot_product":
            scale = torch.sum(
                superposed_embeds * context_vector.unsqueeze(-1),
                dim=-1,
                keepdim=True,
            )
            collapsed_embeds = base_embeds + scale * superposed_embeds
        elif self.collapse_mode == "cosine_similarity":
            scale = F.cosine_similarity(
                superposed_embeds, context_vector.unsqueeze(-1), dim=-1
            ).unsqueeze(-1)
            collapsed_embeds = base_embeds + scale * superposed_embeds
        elif self.collapse_mode == "gated":
            gate = torch.sigmoid(context_vector)
            collapsed_embeds = (
                base_embeds + gate.unsqueeze(-1) * superposed_embeds
            )
        elif self.collapse_mode == "concat_linear":
            concatenated = torch.cat([base_embeds, superposed_embeds], dim=-1)
            collapsed_embeds = self.linear_transform(concatenated)
        else:
            raise ValueError("Invalid collapse mode selected")

        return collapsed_embeds


# # Example Usage
# vocab_size = 10000
# dim = 512

# model = QFTSPEmbedding(vocab_size, dim)
# x = torch.randint(0, vocab_size, (1, 10))
# context_vector = torch.rand(1, 10)

# # Test different collapse modes
# for mode in ['weighted_sum', 'dot_product', 'cosine_similarity', 'gated', 'concat_linear']:
#     embeddings = model(x, context_vector, collapse_mode=mode)
#     print(f"Collapse mode: {mode}, Embeddings shape: {embeddings.shape}")
