from torch import Tensor, nn


def add_norm(x, dim: int, residual: Tensor):
    """_summary_

    Args:
        x (_type_): _description_
        dim (int): _description_
        residual (Tensor): _description_

    Returns:
        _type_: _description_


    Example:
    x = torch.randn(1, 10, 10)
    y = torch.randn(1, 10, 10)
    model = add_norm(x, 10, y)
    print(model)
    """
    layer = nn.Sequential(nn.LayerNorm(dim))
    return layer(x) + residual
