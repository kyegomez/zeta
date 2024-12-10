from typing import Optional
import torch
from torch.optim.optimizer import Optimizer
from loguru import logger
import warnings


class Muon(Optimizer):
    """
    Implementation of the Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

    Muon optimizes 2D parameters of neural networks by taking SGD-momentum updates and
    applying a Newton-Schulz iteration as a post-processing step. It's particularly
    effective for transformer architectures when applied to Q, K, V parameters separately.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0.9)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        nesterov (bool): enables Nesterov momentum (default: True)
        ns_steps (int): number of Newton-Schulz iteration steps (default: 5)
        eps (float): term added to norm for numerical stability (default: 1e-7)
        device (str): device to use for computations ('cuda', 'cpu', or None) (default: None)
        dtype (torch.dtype): data type for Newton-Schulz iterations (default: torch.bfloat16)

    Example:
        >>> model = TransformerModel()
        >>> # Separate Q,K,V parameters for better performance
        >>> qkv_params = []
        >>> other_params = []
        >>> for name, param in model.named_parameters():
        >>>     if any(x in name for x in ['query', 'key', 'value']):
        >>>         qkv_params.append(param)
        >>>     else:
        >>>         other_params.append(param)
        >>> optimizer = Muon(qkv_params, lr=0.001)
        >>> adam = torch.optim.AdamW(other_params, lr=0.001)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(
                f"ns_steps must be a positive integer, got: {ns_steps}"
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        super().__init__(params, defaults)

        # Newton-Schulz coefficients (tuned as per the paper)
        self.ns_coeffs = (3.4445, -4.7750, 2.0315)

        logger.info(
            f"Initialized Muon optimizer with lr={lr}, momentum={momentum}, "
            f"nesterov={nesterov}, ns_steps={ns_steps}"
        )

        # Validate parameters are 2D
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    if p.dim() != 2:
                        warnings.warn(
                            f"Found parameter with {p.dim()} dimensions. "
                            "Muon is designed for 2D parameters only. "
                            "Consider using AdamW for non-2D parameters."
                        )

    @torch.no_grad()
    def _newton_schulz(
        self,
        G: torch.Tensor,
        steps: int,
        eps: float,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Implements the Newton-Schulz matrix iteration for orthogonalization.

        Args:
            G: Input matrix to orthogonalize
            steps: Number of Newton-Schulz iteration steps
            eps: Small constant for numerical stability
            device: Computation device
            dtype: Data type for computations

        Returns:
            Orthogonalized matrix
        """
        assert G.ndim == 2, f"Input matrix must be 2D, got {G.ndim}D"

        # Move to specified device if needed
        if device is not None and G.device.type != device:
            G = G.to(device)

        # Convert to specified dtype if needed
        if dtype is not None:
            G = G.to(dtype)

        a, b, c = self.ns_coeffs
        X = G.clone()

        # Initial normalization
        X /= X.norm() + eps

        # Handle non-square matrices
        transposed = False
        if X.size(0) > X.size(1):
            X = X.T
            transposed = True

        # Newton-Schulz iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        # Restore original shape if needed
        if transposed:
            X = X.T

        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Optional[float]: The loss value returned by the closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            device = group["device"]
            dtype = group["dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.dim() != 2:
                    logger.warning(
                        f"Skipping {p.dim()}D parameter in Muon update"
                    )
                    continue

                grad = p.grad

                # Handle weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Get or initialize momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)

                # Apply Nesterov momentum if enabled
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Apply Newton-Schulz orthogonalization
                update = self._newton_schulz(
                    update, steps=ns_steps, eps=eps, device=device, dtype=dtype
                )

                # Update parameters
                p.add_(update, alpha=-group["lr"])

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """
        Resets the gradients of all optimized parameters.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
        """
        super().zero_grad(set_to_none=set_to_none)

    def get_momentum_buffer(
        self, param: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Returns the momentum buffer associated with a parameter.

        Args:
            param: The parameter whose momentum buffer to return

        Returns:
            The momentum buffer or None if not initialized
        """
        state = self.state[param]
        return state.get("momentum_buffer", None)


# import torch
# import torch.nn as nn
# from muon import Muon  # Assuming muon.py contains our implementation

# # Simple transformer layer
# class SimpleTransformer(nn.Module):
#     def __init__(self, d_model=256):
#         super().__init__()
#         self.query = nn.Linear(d_model, d_model)
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
#         self.output = nn.Linear(d_model, d_model)

#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)

#         # Simple attention
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
#         attn = torch.softmax(scores, dim=-1)
#         out = torch.matmul(attn, v)
#         return self.output(out)

# # Create model
# model = SimpleTransformer()

# # Separate parameters for different optimizers
# muon_params = []
# other_params = []

# for name, param in model.named_parameters():
#     if any(x in name for x in ['query', 'key', 'value']):
#         muon_params.append(param)
#     else:
#         other_params.append(param)

# # Create optimizers
# muon_opt = Muon(muon_params, lr=0.001)
# adam_opt = torch.optim.AdamW(other_params, lr=0.001)

# # Training loop example
# batch_size, seq_len, d_model = 32, 16, 256
# x = torch.randn(batch_size, seq_len, d_model)
# target = torch.randn(batch_size, seq_len, d_model)

# for step in range(10):
#     # Zero gradients
#     muon_opt.zero_grad()
#     adam_opt.zero_grad()

#     # Forward pass
#     output = model(x)
#     loss = nn.MSELoss()(output, target)

#     # Backward pass
#     loss.backward()

#     # Update parameters
#     muon_opt.step()
#     adam_opt.step()

#     print(f"Step {step}, Loss: {loss.item():.4f}")
