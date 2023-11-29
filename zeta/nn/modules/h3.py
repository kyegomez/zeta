import torch
import torch.nn as nn

class DiagonalSSM(nn.Module):
    """DiagonalSSM is a module that implements the Diagonal SSM operation.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, dim):
        super().__init__()
        # A diagonal matrix represented as a vector for ease of multiplication
        self.diag = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Forward

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Multiplication with a diagonal matrix can be done element-wise
        return x * self.diag

class ShiftSSM(nn.Module):
    """ShiftSSM is a module that implements the Shift SSM operation.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, dim):
        super().__init__()
        # A shift matrix operation
        self.dim = dim

    def forward(self, x):
        """Forward pass of the module.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Shift the last dimension of x by one
        return torch.cat((x[..., -1:], x[..., :-1]), dim=-1)

class H3Layer(nn.Module):
    """H3Layer is a layer that implements the H3 associative memory model.
    
    
    Attributes:
        dim (int): The dimensionality of the input and output tensors.
    
    Methods:
        forward(x): Performs a forward pass through the layer.
        
    Examples:
        >>> import torch
        >>> from zeta.nn.modules.h3 import H3Layer
        >>> x = torch.randn(1, 512, 1024)
        >>> layer = H3Layer(512)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([1, 512, 1024])
    """
    def __init__(self, dim: int):
        super().__init__()
        self.diagonal_ssm = DiagonalSSM(dim)
        self.shift_ssm = ShiftSSM(dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply Shift SSM to k
        k = self.shift_ssm(k)
        
        # Element-wise multiplication for associative recall
        combined = q * k
        
        # Apply Diagonal SSM to combined tensor
        output = self.diagonal_ssm(combined) * v
        
        return output

# # Example usage:
# batch_size, seq_len, dim = 32, 40, 512
# x = torch.rand(batch_size, seq_len, dim)
# h3_layer = H3Layer(dim)
# output = h3_layer(x)
# print(output.shape)  # Expected shape: (batch_size, seq_len, dim)
