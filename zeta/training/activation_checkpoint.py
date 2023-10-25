from functools import partial

import torch
from accelerate import Accelerator
import typing
import functools

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
except:
    # let's patch the error.
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper

    def lambda_auto_wrap_policy(
        module: torch.nn.Module,
        recurse: bool,
        unwrapped_params: int,
        lambda_fn: typing.Callable,
    ) -> bool:
        """
        A convenient auto wrap policy to wrap submodules based on an arbitrary user
        function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
        a `wrapper_cls` unit.

        Return if a module should be wrapped during auto wrapping.

        The first three parameters are required by :func:`_recursive_wrap`.

        Args:
        module (nn.Module):
            The module to be considered in this decision.
        recurse (bool):
            Indicate if this is called to make a decision on whether we
            should recurse down a subgraph of the module structure.
            If False, it means this function is called to make a decision
            on whether we should wrap the said module.
        unwrapped_params (int):
            The number of parameters yet to be wrapped in this module.

        lambda_fn (Callable[nn.Module] -> bool):
            If this returns ``True``, this module will be wrapped by
            wrapper_cls individually.
        """
        if recurse:
            # always recurse
            return True
        else:
            # if not recursing, decide whether we should wrap for the leaf node or reminder
            return lambda_fn(module)

    def apply_activation_checkpointing_wrapper(
        model,
        checkpoint_wrapper_fn=torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper,
        check_fn=lambda _: True,
    ):
        """
        Applies :func:`checkpoint_wrapper` to modules within `model` based on a user-defined
        configuration. For each module within `model`, the `check_fn` is used to decide
        whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

        Note::
            This function modifies `model` in place and replaces appropriate layers with
            their checkpoint-wrapped modules.
        Note::
            This function will not wrap the overall root module. If this is needed, please directly use
            :class:`CheckpointWrapper`.
        Usage::
            model = nn.Sequential(
                nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
            )
            check_fn = lambda l: isinstance(l, nn.Linear)
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
        Args:
            module (nn.Module):
                The model who's submodules (or self) should be wrapped with activation checkpointing.
            checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
                A `Callable` which will wrap modules
            check_fn (Optional[Callable[nn.Module, nn.Module]])
                A lambda function which will be passed current layer and returns
                ``True`` or ``False`` depending on whether input layer should be wrapped.
        Returns: None (`model` is modified inplace)
        """
        # TODO: Importing inside function to avoid circular import issue between FSDP and
        # checkpoint_wrapper. This can be resolved once wrap() APIs are decoupled from FSDP code.
        from torch.distributed.fsdp.wrap import _recursive_wrap

        return _recursive_wrap(
            module=model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy, lambda_fn=check_fn
            ),
            wrapper_cls=checkpoint_wrapper_fn,
            ignored_modules=set(),
            ignored_params=set(),
            only_wrap_children=True,
        )

    setattr(
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper,
        "apply_activation_checkpointing",
        apply_activation_checkpointing_wrapper,
    )
    apply_activation_checkpointing = apply_activation_checkpointing_wrapper


def activation_checkpointing(
    model: torch.nn.Module,
    offload_to_cpu: bool = False,
    accelerator: Accelerator = None,
    TransformerBlock=None,
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
