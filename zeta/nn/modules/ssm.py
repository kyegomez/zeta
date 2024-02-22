import torch
import torch.nn as nn
import torch.nn.functional as F

from zeta.nn.modules.p_scan import pscan


def selective_scan(x, delta, A, B, C, D):
    """
    Perform selective scan operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    """
    Perform selective scan sequence operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).
        dim_inner (int): Inner dimension size.
        d_state (int): State dimension size.

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    h = torch.zeros(
        x.size(0),
        dim_inner,
        d_state,
        device=deltaA.device,
    )  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        """
        Initializes the SSM module.

        Args:
            in_features (int): The size of the input features.
            dt_rank (int): The rank of the dt projection.
            dim_inner (int): The inner dimension of the dt projection.
            d_state (int): The dimension of the state.

        """
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        """
        Performs forward pass of the SSM module.

        Args:
            x (torch.Tensor): The input tensor.
            pscan (bool, optional): Whether to use selective_scan or selective_scan_seq. Defaults to True.

        Returns:
            torch.Tensor: The output tensor.

        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan:
            y = selective_scan(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq(x, delta, A, B, C, D)

        return y
