from torch.optim.optimizer import Optimizer


class GradientEquilibrum(Optimizer):
    """
    Gradient Equilibrum optimizer

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        max_iterations (int, optional): maximum number of iterations to find equilibrium
        tol (float, optional): tolerance for equilibrium
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = GradientEquilibrum(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        max_iterations: int = 1000,
        tol=1e-7,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            max_iterations=max_iterations,
            tol=tol,
            weight_decay=weight_decay,
        )
        super(GradientEquilibrum, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Step function for Gradient Equilibrum optimizer

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (float): loss value


        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group["weight_decay"] != 0:
                    grad.add(p.data, alpha=group["weight_decay"])

                # Gradient Equilibrium
                equilibrum_grad = grad - grad.mean()
                p.data -= group["lr"] * equilibrum_grad
        return loss

    def clip_grad_value(self, clip_value):
        """
        CLIp gradient value


        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data.clamp_(-clip_value, clip_value)

    def add_weight_decay(self, weight_decay):
        """
        Add weight decay to the optimizer


        """
        for group in self.param_groups:
            group["weight_decay"] = weight_decay

    def state_dict(self):
        return {
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.param_groups = state_dict["param_groups"]
        self.statet = state_dict["state"]
