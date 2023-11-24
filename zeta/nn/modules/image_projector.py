import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePatchCreatorProjector(nn.Module):
    """
    Image Patch Creator and Projector Layer.

    This layer dynamically creates and projects image patches suitable for
    feeding into a transformer decoder. It is designed to handle input tensors
    of arbitrary shape and outputs a tensor of shape (B, SEQLEN, Dimension).

    Attributes:
        max_patch_size (int): The maximum size of each image patch.
        embedding_dim (int): The dimension of the output embeddings.
    """

    def __init__(self, max_patch_size, embedding_dim):
        """
        Initializes the ImagePatchCreatorProjector.

        Args:
            max_patch_size (int): The maximum size of each image patch.
            embedding_dim (int): The dimension of the output embeddings.
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_dim = embedding_dim
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (max_patch_size, max_patch_size)
        )
        self.projection = None

    def forward(self, x):
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): The input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: The output tensor with shape (B, SEQLEN, Dimension).
        """
        try:
            B, C, H, W = x.shape
            dynamic_patch_size = self.calculate_dynamic_patch_size(H, W)
            self.projection = nn.Linear(
                dynamic_patch_size * dynamic_patch_size * C, self.embedding_dim
            )

            x = self.create_patches(x, dynamic_patch_size)
            x = self.adaptive_pool(x)
            x = x.view(B, -1, dynamic_patch_size * dynamic_patch_size * C)
            x = self.projection(x)

            return x
        except Exception as e:
            # Handle exceptions and potentially log them
            print(f"Error during forward pass: {e}")
            return None

    def calculate_dynamic_patch_size(self, H, W):
        """
        Calculate dynamic patch size based on the dimensions of the input image.

        Args:
            H (int): Height of the input image.
            W (int): Width of the input image.

        Returns:
            int: Calculated patch size.
        """
        # Example logic; this can be adjusted based on specific requirements
        return min(H, W, self.max_patch_size)

    def create_patches(self, x, patch_size):
        """
        Create image patches from the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            patch_size (int): Size of each patch.

        Returns:
            torch.Tensor: Tensor with created patches.
        """
        B, C, H, W = x.shape
        x = x.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        x = x.contiguous().view(B, -1, patch_size, patch_size, C)
        x = (
            x.permute(0, 1, 4, 2, 3)
            .contiguous()
            .view(B, -1, patch_size, patch_size)
        )
        return x


# # Example Usage
# # Initialize the layer
# patch_projector = ImagePatchCreatorProjector(max_patch_size=16, embedding_dim=768)

# # Example input tensor (randomly generated for demonstration)
# input_tensor = torch.randn(1, 3, 64, 64)  # Shape: [B, C, H, W]

# # Forward pass
# output_tensor = patch_projector(input_tensor)
# print(
#     f"Output Shape: {output_tensor.shape if output_tensor is not None else 'Error in processing'}"
# )
