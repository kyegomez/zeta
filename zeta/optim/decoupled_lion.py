import logging
import math
from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)


class DecoupledLionW(Optimizer):
    """
    DecoupledLionW is an optimizer designed to improve training performance and convergence for deep learning models.
    It is an extension of the Lion optimizer, incorporating decoupled weight decay and a momentum-based update rule.
    The optimizer utilizes the Adam-like update rule, where the weight decay is applied separately from the gradient update.
    The update rule consists of three steps: weight decay, momentum update, and momentum decay.
    Weight decay reduces the magnitude of the model's weights, preventing overfitting and improving generalization.
    The momentum update is an interpolation between the current gradient and the previous momentum state, allowing for faster convergence and smoother optimization.
    Momentum decay gradually reduces the momentum term over time, preventing it from becoming too large and destabilizing the optimization process.
    The optimizer supports both single-node and multi-node distributed training, enabling efficient training on parallel computing environments.
    It provides various metric functions to track the optimization process, such as L2 norm of moments, parameters, updates, and gradients, as well as cosine similarity between updates and gradients.
    The optimizer allows reporting per-parameter metrics to analyze the behavior of individual model parameters during training.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)

    Example:

        >>> import torch
        >>> from zeta.optim import DecoupledLionW
        >>> from zeta.training import get_lr_scheduler_with_warmup
        >>> from zeta.training import Accelerator
        >>> from torch.utils.data import DataLoader
        >>> from transformers import AdamW
        >>> from transformers import AutoTokenizer
        >>> from transformers import AutoModelForSequenceClassification
        >>> from transformers import default_data_collator
        >>> from transformers import TrainingArguments
        >>> from transformers import Trainer
        >>> from datasets import load_dataset
        >>>
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        >>> dataset = load_dataset("glue", "mrpc")
        >>> dataset = dataset.remove_columns(["idx", "label"])
        >>> dataset = dataset.rename_column("sentence1", "premise")
        >>> dataset = dataset.rename_column("sentence2", "hypothesis")
        >>> dataset = dataset.map(lambda x: tokenizer(x["premise"], x["hypothesis"], truncation=True, padding="max_length"), batched=True)
        >>> dataset = dataset.map(lambda x: {"labels": x["label"]}, batched=True)
        >>> dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        >>> train_dataset, eval_dataset = dataset["train"], dataset["validation"]
        >>> data_collator = default_data_collator
        >>> training_args = TrainingArguments(
        ...     output_dir="./results",
        ...     num_train_epochs=3,
        ...     per_device_train_batch_size=32,
        ...     per_device_eval_batch_size=32,
        ...     warmup_steps=500,
        ...     weight_decay=0.01,
        ...     logging_dir="./logs",
        ...     logging_steps=10,
        ...     evaluation_strategy="epoch",
        ...     save_strategy="epoch",
        ...     load_best_model_at_end=True,
        ...     metric_for_best_model="accuracy",
        ...     greater_is_better=True,
        ... )
        >>> optimizer = DecoupledLionW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
        >>> lr_scheduler = get_lr_scheduler_with_warmup(optimizer, "linear", num_warmup_steps=500, max_train_steps=training_args.max_steps)
        >>> accelerator = Accelerator()
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     data_collator=data_collator,
        ...     tokenizer=tokenizer,
        ...     optimizers=(optimizer, None),
        ...     lr_schedulers=(lr_scheduler, None),
        ...     compute_metrics=lambda p: {"accuracy": p["eval_accuracy"]},
        ... )
        >>> trainer.train()


    """

    metric_functions = {
        "l2_norm/moment": (
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(
                optim_state["exp_avg"]
            )
        ),
        "l2_norm/param": (
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(
                param.data
            )
        ),
        "l2_norm/update": (
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(
                step_tensor
            )
        ),
        "l2_norm/grad": (
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(
                param.grad
            )
        ),
        "cosine/update_grad": lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
            param.grad.flatten(), step_tensor.flatten(), dim=0
        ),
        "cosine/moment_grad": lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
            param.grad.flatten(), optim_state["exp_avg"].flatten(), dim=0
        ),
    }

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise Exception(f"Invalid LR: {lr}. LR must be > 0")
        if not all([0.0 <= beta <= 1.0 for beta in betas]):
            raise Exception(
                f"Invalid beta values: {betas}. All betas must be between 0"
                " and 1."
            )
        if weight_decay >= 1e-3:
            log.warning(
                f"You are using a high value of `weight_decay={weight_decay}`"
                " for the `DecoupledLionW` optimizer. Are you sure you want to"
                " do this? Your model's weights will be multiplied by"
                f" {1.0 - weight_decay} on every step!"
            )

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}

        super().__init__(params, defaults)

        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @staticmethod
    def lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2) -> None:
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(
                lambda p: p.grad is not None and p.requires_grad,
                group["params"],
            ):
                grad, lr, initial_lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["initial_lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2)

        return loss

    def pre_reduce_metrics(self, optimizer_metrics):
        metrics = optimizer_metrics.keys()
        metrics = sorted(
            metrics, key=lambda metric: 0 if "l2_norm" in metric else 1
        )
        for metric in metrics:
            if metric.startswith("l2_norm"):
                optimizer_metrics[metric] = optimizer_metrics[metric] ** 2
            elif metric.startswith("cosine"):
                _, vectors, layer = tuple(metric.split("/"))
                A, B = tuple(vectors.split("_"))
                A_rank_subset_norm = math.sqrt(
                    optimizer_metrics[f"l2_norm/{A}/{layer}"]
                )
                B_rank_subset_norm = math.sqrt(
                    optimizer_metrics[f"l2_norm/{B}/{layer}"]
                )
                optimizer_metrics[metric] *= (
                    A_rank_subset_norm * B_rank_subset_norm
                )

        return optimizer_metrics

    def report_per_parameter_metrics(
        self, param: torch.Tensor, name: str, optimizer_metrics: dict
    ):
        lr = self.param_groups[0]["lr"]
        weight_decay = self.param_groups[0]["weight_decay"]
        initial_lr = self.param_groups[0]["initial_lr"]

        beta1, _ = self.param_groups[0]["betas"]

        if param in self.state:
            param_optim_state = self.state[param]
            step_tensor = (
                param_optim_state["exp_avg"]
                .clone()
                .lerp_(param.grad, 1 - beta1)
                .sign_()
                .mul_(lr)
            )

            decay_factor = (lr / initial_lr) if initial_lr else 1.0

            step_tensor.add_(param, alpha=-weight_decay * decay_factor)

            for metric in self.metric_functions:
                optimizer_metrics[f"{metric}/{name}"] = self.metric_functions[
                    metric
                ](param, param_optim_state, step_tensor)

        return optimizer_metrics
