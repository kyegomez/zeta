import torch
from torch import Tensor, nn


class Laser(nn.Module):
    """
    Layer Selective Rank Reduction (LASER) is a module that replaces specific weight matrices
    in a Transformer model by their low-rank approximations for both 2D and 3D tensors.

    Attributes:
        rank_fraction (float): Fraction of the maximum rank to preserve in the approximation (value between 0 and 1).

    Examples:
    # Example usage
    d = 512  # Dimension of the weight matrix
    # Example weight matrix - can be a 2D or 3D tensor
    W_2d = torch.randn(d, d)  # 2D tensor
    W_3d = torch.randn(10, d, d)  # 3D tensor with a batch size of 10
    rank_fraction = 0.9  # Fraction of the rank to preserve

    # Create the LASER module
    laser = LASER(rank_fraction)

    # Apply LASER to 2D and 3D tensors
    W_2d_low_rank = laser(W_2d)
    W_3d_low_rank = laser(W_3d)

    print(W_2d_low_rank.shape)  # The shape of the approximated matrix will be the same as the original 2D matrix
    print(W_3d_low_rank.shape)  # The shape of the approximated matrices will be the same as the original 3D tensor

    """

    def __init__(self, rank_fraction):
        """
        Args:
            rank_fraction (float): Fraction of the maximum rank to preserve in the approximation.
        """
        super().__init__()
        assert 0 <= rank_fraction < 1, "rank_fraction must be between 0 and 1."
        self.rank_fraction = rank_fraction

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the low-rank approximation to the weight matrix or batch of matrices.

        Args:
            x (Tensor): The weight matrix or batch of matrices to be approximated.

        Returns:
            torch.Tensor: The approximated weight matrix or batch of matrices with reduced rank.
        """
        # Handle 3D tensors
        if x.ndim == 3:
            # Process each matrix in the batch individually
            W_approx = torch.stack([self.low_rank_approximation(m) for m in x])
        else:  # Handle 2D tensors
            W_approx = self.low_rank_approximation(x)

        return W_approx

    def low_rank_approximation(self, matrix: Tensor) -> Tensor:
        """
        Helper function to perform low-rank approximation on a 2D matrix.

        Args:
            matrix (Tensor): The 2D matrix to be approximated.

        Returns:
            torch.Tensor: The approximated 2D matrix with reduced rank.
        """
        U, S, V = torch.svd(matrix)
        max_rank = min(matrix.size())
        approx_rank = int(self.rank_fraction * max_rank)
        U_r = U[:, :approx_rank]
        S_r = S[:approx_rank]
        V_r = V[:, :approx_rank]
        W_approx = torch.mm(U_r, torch.mm(torch.diag(S_r), V_r.t()))
        return W_approx
