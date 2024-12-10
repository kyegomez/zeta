import math
from typing import List, Dict, Optional, Tuple, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer


class FastAdaptiveOptimizer(Optimizer):
    """
    FastAdaptiveOptimizer implements a novel optimization algorithm that aims to be
    faster than Adam while maintaining accuracy and potentially finding local minima more quickly.

    It combines elements from Adam, RMSprop, and SGD with momentum, while adding
    features like warmup and the option to switch between adaptive and non-adaptive modes.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        momentum (float, optional): momentum factor (default: 0.9)
        adaptive_lr (bool, optional): whether to use adaptive learning rate
            (default: True)
        warmup_steps (int, optional): number of warmup steps (default: 1000)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.9,
        adaptive_lr: bool = True,
        warmup_steps: int = 1000,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            momentum=momentum,
            adaptive_lr=adaptive_lr,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(FastAdaptiveOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state: Dict) -> None:
        super(FastAdaptiveOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Optional[float]: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            momentum_buffers = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "FastAdaptiveOptimizer does not support sparse gradients"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    momentum_buffers.append(state["momentum_buffer"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    state["step"] += 1
                    state_steps.append(state["step"])

            beta1, beta2 = group["betas"]
            self._update_params(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                momentum_buffers,
                state_steps,
                group["amsgrad"],
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["momentum"],
                group["adaptive_lr"],
                group["warmup_steps"],
            )

        return loss

    def _update_params(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        momentum_buffers: List[Tensor],
        state_steps: List[int],
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        momentum: float,
        adaptive_lr: bool,
        warmup_steps: int,
    ) -> None:
        """
        Function to update parameters.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            momentum_buffer = momentum_buffers[i]
            step = state_steps[i]

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                torch.maximum(
                    max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i]
                )
                denom = max_exp_avg_sqs[i].sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            if adaptive_lr:
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            else:
                step_size = lr

            # Apply warmup
            if step < warmup_steps:
                step_size *= step / warmup_steps

            # Compute the update
            if adaptive_lr:
                update = exp_avg / denom
            else:
                update = exp_avg

            # Apply momentum and update parameters
            momentum_buffer.mul_(momentum).add_(update)
            param.add_(momentum_buffer, alpha=-step_size)
