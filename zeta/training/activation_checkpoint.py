
from functools import partial

import torch
from accelerate import Accelerator


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)


def activation_checkpointing(
    model: torch.nn.Module,
    offload_to_cpu: bool = False,
    accelerator: Accelerator = None,
    TransformerBlock = None
):
    """
    Apply activation checkpointing to a model.

    Args:
        model (Module): The model to which to apply activation checkpointing.
        offload_to_cpu (bool, optional): Whether to offload the activations to CPU. Defaults to False.
        accelerator (Accelerator, optional): The Accelerate library accelerator. Defaults to None.
    """
    if accelerator is not None:
        accelerator.print("Using activation checkpointing")
    def check_fn(submodule):
        return isinstance(submodule, TransformerBlock)
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=offload_to_cpu,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
