import torch
from accelerate import Accelerator

from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def get_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    max_train_steps: int,
    grad_accumulate_every: int = 1,
    accelerator: Accelerator = None,
):
    """
    Get a learning rate scheduler with warmup.

    Args:
        optimizer (Optimizer): The optimizer for which to create the learning rate scheduler.
        scheduler_type (str): The type of learning rate scheduler to create, either "linear" or "cosine".
        num_warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        max_train_steps (int): The maximum number of training steps.
        grad_accumulate_every (int, optional): The gradient accumulation factor. Defaults to 1.
        accelerator (Accelerator, optional): The Accelerate library accelerator. Defaults to None.

    Returns:
        The learning rate scheduler with warmup.

    Raises:
        ValueError: If scheduler_type is not "linear" or "cosine".
    """
    NUM_WARMUP_STEPS = num_warmup_steps
    GRADIENT_ACCUMULATE_EVERY = grad_accumulate_every
    if accelerator is not None:
        accelerator.print(f"Using {scheduler_type} lr scheduler")
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
        )
    else:
        raise ValueError(
            "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}"
            .format(scheduler_type)
        )
