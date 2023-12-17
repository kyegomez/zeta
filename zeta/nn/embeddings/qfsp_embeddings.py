import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumSuperpositionEmbeddings(nn.Module):
    """
    QuantumSuperpositionEmbeddings with multiple collapse mechanisms.

    This module allows for different ways of collapsing the superposition of embeddings,
    based on the provided context and selected mechanism.
    """

    def __init__(self, vocab_size, embed_dim):
        super(QuantumSuperpositionEmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.base_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.superposed_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear_transform = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, input_ids, context_vector, collapse_mode="weighted_sum"):
        base_embeds = self.base_embeddings(input_ids)
        superposed_embeds = self.superposed_embeddings(input_ids)

        if collapse_mode == "weighted_sum":
            collapsed_embeds = (
                base_embeds + context_vector.unsqueeze(-1) * superposed_embeds
            )
        elif collapse_mode == "dot_product":
            scale = torch.sum(
                superposed_embeds * context_vector.unsqueeze(-1),
                dim=-1,
                keepdim=True,
            )
            collapsed_embeds = base_embeds + scale * superposed_embeds
        elif collapse_mode == "cosine_similarity":
            scale = F.cosine_similarity(
                superposed_embeds, context_vector.unsqueeze(-1), dim=-1
            ).unsqueeze(-1)
            collapsed_embeds = base_embeds + scale * superposed_embeds
        elif collapse_mode == "gated":
            gate = torch.sigmoid(context_vector)
            collapsed_embeds = (
                base_embeds + gate.unsqueeze(-1) * superposed_embeds
            )
        elif collapse_mode == "concat_linear":
            concatenated = torch.cat([base_embeds, superposed_embeds], dim=-1)
            collapsed_embeds = self.linear_transform(concatenated)
        else:
            raise ValueError("Invalid collapse mode selected")

        return collapsed_embeds


# # Example Usage
# vocab_size = 10000
# embed_dim = 512

# model = QuantumSuperpositionEmbeddings(vocab_size, embed_dim)
# input_ids = torch.randint(0, vocab_size, (1, 10))
# context_vector = torch.rand(1, 10)

# # Test different collapse modes
# for mode in ['weighted_sum', 'dot_product', 'cosine_similarity', 'gated', 'concat_linear']:
#     embeddings = model(input_ids, context_vector, collapse_mode=mode)
#     print(f"Collapse mode: {mode}, Embeddings shape: {embeddings.shape}")
