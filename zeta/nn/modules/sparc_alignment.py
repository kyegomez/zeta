import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SparseFineGrainedContrastiveAlignment(nn.Module):
    def __init__(
        self,
        vision_adapter: nn.Module,
        text_adapter: nn.Module,
        hidden_dim: int,
        tau: float = 0.07,
    ):
        super(SparseFineGrainedContrastiveAlignment, self).__init__()
        self.vision_adapter = vision_adapter
        self.text_adapter = text_adapter
        self.hidden_dim = hidden_dim
        self.tau = tau

    def forward(
        self, image_patches: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        # Assume image_patches: [b, c, h, w] and text_tokens: [b, s, d] are already encoded

        # Flatten image patches for easier processing
        b, c, h, w = image_patches.shape
        image_patches = rearrange(
            image_patches, "b c h w -> b (h w) c"
        )  # shape: [b, hw, c]

        # Apply adapters
        image_patches = self.vision_adapter(image_patches)  # shape: [b, hw, d]
        text_tokens = self.text_adapter(text_tokens)  # shape: [b, s, d]

        # Compute global embeddings
        global_image_embedding = self.vision_adapter(
            F.adaptive_avg_pool2d(
                rearrange(image_patches, "b p d -> b d p"), (1, 1)
            ).squeeze(-1)
        )  # shape: [b, d]
        global_text_embedding = self.text_adapter(
            F.adaptive_avg_pool1d(
                rearrange(text_tokens, "b s d -> b d s"), 1
            ).squeeze(-1)
        )  # shape: [b, d]

        # Global contrastive loss
        global_loss = self.global_contrastive_loss(
            global_image_embedding, global_text_embedding
        )

        # Fine-grained alignment
        fine_grained_loss = self.fine_grained_alignment(
            image_patches, text_tokens
        )

        # Overall loss
        overall_loss = global_loss + fine_grained_loss

        return overall_loss

    def global_contrastive_loss(
        self,
        global_image_embedding: torch.Tensor,
        global_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        b, d = global_image_embedding.shape
        sim_matrix = (
            F.cosine_similarity(
                global_image_embedding.unsqueeze(1),
                global_text_embedding.unsqueeze(0),
                dim=-1,
            )
            / self.tau
        )
        labels = torch.arange(b).long().to(global_image_embedding.device)
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_t = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_i + loss_t) / 2
        return loss

    def fine_grained_alignment(
        self, image_patches: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        b, hw, d = image_patches.shape
        _, s, _ = text_tokens.shape

        # Compute similarity matrix
        sim_matrix = torch.einsum(
            "bpd,bsd->bps", image_patches, text_tokens
        )  # shape: [b, hw, s]

        # Min-max normalization
        sim_matrix = (sim_matrix - sim_matrix.min(dim=1, keepdim=True)[0]) / (
            sim_matrix.max(dim=1, keepdim=True)[0]
            - sim_matrix.min(dim=1, keepdim=True)[0]
            + 1e-8
        )

        # Sparsification
        sigma = 1 / hw
        sim_matrix[sim_matrix < sigma] = 0

        # Compute alignment weights
        alignment_weights = F.normalize(
            sim_matrix, p=1, dim=1
        )  # shape: [b, hw, s]

        # Compute language-grouped vision embeddings
        language_grouped_vision_embeddings = torch.einsum(
            "bps,bpd->bsd", alignment_weights, image_patches
        )  # shape: [b, s, d]

        # Fine-grained contrastive loss
        fine_grained_loss = self.fine_grained_contrastive_loss(
            language_grouped_vision_embeddings, text_tokens
        )

        return fine_grained_loss

    def fine_grained_contrastive_loss(
        self,
        language_grouped_vision_embeddings: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        b, s, d = language_grouped_vision_embeddings.shape
        sim_matrix = (
            F.cosine_similarity(
                language_grouped_vision_embeddings.unsqueeze(2),
                text_tokens.unsqueeze(1),
                dim=-1,
            )
            / self.tau
        )
        labels = (
            torch.arange(s).long().to(language_grouped_vision_embeddings.device)
        )
        loss_c = F.cross_entropy(sim_matrix.permute(0, 2, 1), labels)
        loss_t = F.cross_entropy(sim_matrix, labels)
        loss = (loss_c + loss_t) / 2
        return loss


# # Example usage:
# # Assuming vision_adapter and text_adapter are defined elsewhere
# model = SparseFineGrainedContrastiveAlignment(
#     vision_adapter, text_adapter, hidden_dim=768
# )
# image_patches = torch.randn(32, 3, 224, 224)  # Example image batch
# text_tokens = torch.randn(32, 128, 768)  # Example text batch
# loss = model(image_patches, text_tokens)
# print(loss)
