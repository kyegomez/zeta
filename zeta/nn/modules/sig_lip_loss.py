import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLipSigmoidLoss(nn.Module):
    """
    SigmoidLoss is a custom loss function that computes the sigmoid loss between image and text embeddings.

    Args:
        dim (int): The dimension of the embeddings.

    Attributes:
        t_prime (nn.Parameter): The temperature parameter.
        b (nn.Parameter): The bias term.
        dim (int): The dimension of the embeddings.

    Methods:
        forward(img_emb, txt_emb): Computes the sigmoid loss between image and text embeddings.

    """

    def __init__(self, dim: int):
        super(SigLipSigmoidLoss, self).__init__()
        self.t_prime = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.dim = dim

    def forward(self, img_emb, txt_emb):
        """
        Computes the sigmoid loss between image and text embeddings.

        Args:
            img_emb (torch.Tensor): The image embeddings.
            txt_emb (torch.Tensor): The text embeddings.

        Returns:
            torch.Tensor: The computed sigmoid loss.

        Raises:
            AssertionError: If the shape of image and text embeddings are not the same.
            AssertionError: If the embedding dimension is not equal to `self.dim`.

        """
        # Ensure embeddings are of correct shape
        assert (
            img_emb.shape == txt_emb.shape
        ), "Image and text embeddings must have the same shape"
        assert (
            img_emb.shape[2] == self.dim
        ), f"Embedding dimension must be {self.dim}"

        # Get batch size and n
        batch_size, n, _ = img_emb.shape

        # Temperature parameter
        t = torch.exp(self.t_prime)

        # Normalize embeddings
        zimg = F.normalize(img_emb, p=2, dim=2)
        ztxt = F.normalize(txt_emb, p=2, dim=2)

        # Compute logits
        logits = torch.matmul(zimg, ztxt.transpose(1, 2)) * t + self.b

        # Create labels
        labels = 2 * torch.eye(n, device=logits.device).unsqueeze(0).expand(
            batch_size, -1, -1
        ) - torch.ones(batch_size, n, n, device=logits.device)

        # Compute loss
        loss = -torch.sum(F.logsigmoid(labels * logits)) / (batch_size * n)

        return loss


# Example usage
# if __name__ == "__main__":
#     batch_size = 16
#     n = 10
#     dim = 512

#     # Dummy embeddings
#     img_emb = torch.randn(batch_size, n, dim)
#     txt_emb = torch.randn(batch_size, n, dim)

#     # Initialize loss module
#     loss_module = SigmoidLoss(dim)

#     # Compute loss
#     loss = loss_module(img_emb, txt_emb)
#     print("Loss:", loss.item())
