import torch
import torch.nn as nn

class FilmConditioning(nn.Module):
    """
    FilmConditioning module applies feature-wise affine transformations to the input tensor based on conditioning tensor.
    
    Args:
        num_channels (int): Number of channels in the input tensor.
    
    Attributes:
        num_channels (int): Number of channels in the input tensor.
        _projection_add (nn.Linear): Linear layer for additive projection.
        _projection_mult (nn.Linear): Linear layer for multiplicative projection.
        
    Examples:
        >>> conv_filters = torch.randn(10, 3, 32, 32)
        >>> conditioning = torch.randn(10, 3)
        >>> film_conditioning = FilmConditioning(3)
        >>> result = film_conditioning(conv_filters, conditioning)
        >>> print(result.shape)
        torch.Size([10, 3, 32, 32])
    """
    def __init__(
        self,
        num_channels: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_channels = num_channels
        self._projection_add = nn.Linear(
            num_channels,
            num_channels,
        )
        self._projection_mult = nn.Linear(
            num_channels,
            num_channels
        )
        
        nn.init.zeros_(self._projection_add.weight)
        nn.init.zeros_(self._projection_add.bias)
        nn.init.zeros_(self._projection_mult.weight)
        nn.init.zeros_(self._projection_mult.bias)
        
    def forward(
        self,
        conv_filters: torch.Tensor,
        conditioning: torch.Tensor
    ):
        """
        Forward pass of the FilmConditioning module.
        
        Args:
            conv_filters (torch.Tensor): Convolutional filters tensor.
            conditioning (torch.Tensor): Conditioning tensor.
        
        Returns:
            torch.Tensor: Result of applying feature-wise affine transformations to the input tensor.
        """
        assert len(conditioning.shape) == 2
        assert conditioning.shape[1] == self.num_channels, "Number of channels in conditioning tensor must match num_channels"
        assert conv_filters.shape[1] == self.num_channels, "Number of channels in conv_filters tensor must match num_channels"
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)
        
        if len(conv_filters.shape) == 4:
            projected_cond_add = projected_cond_add.unsqueeze(1).unsqueeze(2)
            projected_cond_mult = projected_cond_mult.unsqueeze(1).unsqueeze(2)
        else:
            assert len(conv_filters.shape) == 2
            
        result = (1 + projected_cond_add) * conv_filters + projected_cond_add
        return result
    
