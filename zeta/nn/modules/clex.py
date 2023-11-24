import torch
import torch.nn as nn
from torchdiffeq import odeint

import math


class ODELinear(nn.Module):
    def __init__(self, dim: int, factor, **kwargs):
        super().__init__()
        self.ode_up_proj = nn.Parameter(
            torch.empty(dim // 2, factor * dim).to(torch.float32)
        )
        self.ode_down_proj = nn.Parameter(
            torch.empty(factor * dim, dim // 2).to(torch.float32)
        )
        self.dim = dim
        self.act = torch.nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.ode_up_proj, a=math.sqrt(5))
        nn.init.zeros_(self.ode_down_proj)

    def get_time_embedding(
        self, t, base=10000, device="cuda", dtype=torch.float32
    ):
        if t < 1:
            alpha = 1
        else:
            alpha = 2 * t - 1
        ntk_base = base * alpha ** (self.dim / (self.dim - 2))
        ntk_inv_freq = 1.0 / (
            ntk_base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32).to(device)
                / self.dim
            )
        )
        index = torch.arange(0, self.dim, 2, dtype=torch.float32).to(device)
        delta_ntk_freq = (
            -2
            * index
            / (self.dim - 2)
            * 1
            / (
                base ** (index / self.dim)
                * (alpha ** (index / (self.dim - 2) + 1))
            )
        )
        return delta_ntk_freq.to(device, dtype=dtype), ntk_inv_freq.to(
            device, dtype=dtype
        )

    def forward(self, t, x: torch.Tensor):
        delta_time, time = self.get_time_embedding(
            t, device=x.device, dtype=x.dtype
        )
        x = x + torch.log(time)
        time_embed = delta_time / time
        delta_inv_freq = (
            self.act(x @ self.ode_up_proj.float()) @ self.ode_down_proj.float()
            + time_embed
        )
        return delta_inv_freq


class Clex(nn.Module):
    """
    CLEx: Continuous Rotation Positional Encoding

    Args:
        dim: dimension of the input
        max_position_embeddings: maximum number of positions to be encoded
        rope_scaling: dictionary containing the parameters for the rope scaling
            - max_factor: maximum factor for the rope scaling
            - param_factor: factor for the rope scaling
        base: base for the positional encoding
        device: device for the positional encoding

    Returns:
        positional encoding of the input

    Examples:
        >>> import torch
        >>> from zeta.nn.modules.clex import Clex
        >>> clex = Clex(512, max_position_embeddings=2048, rope_scaling={"max_factor": 100, "param_factor": 100})
        >>> input = torch.randn(1, 1, 512)
        >>> output = clex(input)


    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        rope_scaling=None,
        base=10000,
        device=None,
    ) -> None:
        super().__init__()

        self.max_t = rope_scaling["max_factor"]
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self.proj_func = ODELinear(dim, rope_scaling["param_factor"])
        self.rope_cached = None
        self.max_t_cached = 0
        self.freq_cached = None
        self.time_dt = 0.01
        self.ode_args = {
            "method": "rk4",
            "options": {"step_size": self.time_dt},
        }

    def sample_random_times(self, max_t, device):
        return torch.randint(2, max_t, (1,), dtype=torch.long, device=device)

    def get_random_position_ids(self, n=2048, max=8192):
        positions = torch.randperm(max)[:n].sort().values
        # positions = positions.to(device=device)
        return positions

    def get_continuous_freq(self, time_grid, ex_positions, device):
        solution = odeint(
            self.proj_func,
            torch.log(self.inv_freq.to(device, dtype=torch.float32)),
            time_grid,
            **self.ode_args,
        )
        if time_grid.size(0) == 2:
            scale_inv_freq = torch.exp(solution[1])
            # print(time_grid[1].tolist(), torch.sum(scale_inv_freq).tolist(), torch.sum(self.proj_func.ode_down_proj).tolist())
            freqs = torch.outer(ex_positions.float().squeeze(), scale_inv_freq)
        else:
            scale_inv_freq = torch.exp(solution)
            # freqs = torch.einsum('i, kl -> kil', ex_positions, scale_inv_freq)
            return scale_inv_freq
        embed = torch.cat((freqs, freqs), dim=-1)
        return embed

    def forward(self, device, dtype, seq_len, do_train=False):
        device = self.proj_func.ode_up_proj.device
        scale_factor = seq_len // self.max_position_embeddings
        if do_train:
            t_val = self.sample_random_times(self.max_t + 1, device)[0]
            import math

            sampled_position_ids = self.get_random_position_ids(
                n=seq_len - 2, max=seq_len * t_val - 2
            ).float()
            ex_positions = torch.cat(
                [
                    torch.tensor([0]),
                    (sampled_position_ids + 1) / scale_factor,
                    torch.tensor([seq_len * t_val // scale_factor - 1]),
                ]
            ).to(device, dtype=torch.float32)
        else:
            t_val = (
                scale_factor
                if seq_len % self.max_position_embeddings == 0.0
                else scale_factor + 1
            )
            t_val = t_val if t_val <= self.max_t else self.max_t
            ex_positions = torch.arange(
                0, self.max_position_embeddings * t_val, dtype=torch.float32
            ).to(device)

        if t_val == 1.0:
            scale_inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(ex_positions.float().squeeze(), scale_inv_freq)
            embed = torch.cat((freqs, freqs), dim=-1)
            cos, sin = (
                embed.cos()[None, None, :, :],
                embed.sin()[None, None, :, :],
            )
        elif do_train:
            time_grid = torch.tensor([1.0, t_val]).float().to(device)
            embed = self.get_continuous_freq(time_grid, ex_positions, device)
            cos, sin = (
                embed.cos()[None, None, :, :],
                embed.sin()[None, None, :, :],
            )
        else:
            if t_val > self.max_t_cached:
                if self.freq_cached is None:
                    time_grid = torch.arange(
                        1.0, self.max_t + 1.0, dtype=torch.float32
                    ).to(device)
                    self.freq_cached = self.get_continuous_freq(
                        time_grid, ex_positions, device
                    )
                scale_inv_freq = self.freq_cached[int(t_val - 1.0)]
                freqs = torch.outer(
                    ex_positions.float().squeeze(), scale_inv_freq
                )
                embed = torch.cat((freqs, freqs), dim=-1)
                self.rope_cached = torch.cat(
                    (
                        embed.cos()[None, None, None, :, :],
                        embed.sin()[None, None, None, :, :],
                    ),
                    dim=0,
                )
                self.max_t_cached = t_val
            cos, sin = self.rope_cached

        return torch.cat(
            (
                cos[None, :, :, :seq_len, ...].to(dtype=dtype),
                sin[None, :, :, :seq_len, ...].to(dtype=dtype),
            ),
            dim=0,
        )
