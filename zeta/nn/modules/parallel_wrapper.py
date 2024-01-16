from torch import nn


class Parallel(nn.Module):
    """
    A module that applies a list of functions in parallel and sums their outputs.

    Args:
        *fns: Variable number of functions to be applied in parallel.

    Example:
        >>> fn1 = nn.Linear(10, 5)
        >>> fn2 = nn.Linear(10, 5)
        >>> parallel = Parallel(fn1, fn2)
        >>> input = torch.randn(1, 10)
        >>> output = parallel(input)
    """

    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])
