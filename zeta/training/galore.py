import torch
from torch import nn
from typing import Tuple, Iterable


class GaloreOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_dim: int,
        compact_dim: int,
        params: Iterable[torch.Tensor],
        lr: float = 0.002,
        weight_decay: float = 0.2,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        clip_thresh: float = 1.0,
        precision: str = "amp_bfloat16",
        custom_scalar: int = 65536,
    ) -> None:
        super(GaloreOptimizer, self).__init__(
            params,
            dict(
                lr=lr, weight_decay=weight_decay, beta1=betas[0], beta2=betas[1]
            ),
        )
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.eps = eps
        self.d = clip_thresh
        self.precision = precision
        self.custom_scaler = custom_scalar
        # Initialize the projection and back projection layers
        self.proj = nn.Linear(model_dim, compact_dim).to(device)
        self.back_proj = nn.Linear(compact_dim, model_dim).to(device)
        for group in self.param_groups:
            group["step"] = 1.0
        print("Using StableAdamWUnfused-v1")

    def step(self, closure=None):
        """Performs a single optimization step (parameter update)."""
        if closure is not None:
            closure_result = closure()

        for group in self.param_groups:
            lr = group["lr"]
            group["weight_decay"]
            group["beta1"]
            group["beta2"]
            group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Original gradient
                g = p.grad.data
                if self.precision == "custom_fp16":
                    g = g / self.custom_scaler
                if torch.any(torch.isnan(g) | torch.isinf(g)):
                    continue

                # Projection to compact space
                g_compact = self.proj(g.view(1, -1)).view_as(g)

                # Here you can include the update logic (e.g., Adam, SGD) applied on `g_compact`
                # For simplicity, let's use a simplified update rule directly on the compact representation
                # Note: This is where you'd typically integrate with self.optimizer logic for a real implementation
                # Assuming g_compact has been obtained from the projection of gradients
                lr = group["lr"]

                # Simplified update rule (akin to SGD) in compact space
                update_compact = -lr * g_compact

                # Back-projection to original space for applying the update
                update_original = self.back_proj(
                    update_compact.view(1, -1)
                ).view_as(g)

                # Apply update to the parameters
                p.data.add_(update_original)

            group["step"] += 1

        return closure_result if closure is not None else None
