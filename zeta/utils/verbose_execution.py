import functools
import time
from torch import Tensor, nn
from loguru import logger
from typing import Dict, Any, Callable
import torch.cuda as cuda
from typing import Union


def verbose_execution(
    log_params: bool = False,
    log_gradients: bool = False,
    log_memory: bool = True,
):
    """
    A decorator that adds verbose logging to a PyTorch model's execution.

    Args:
        log_params (bool): Whether to log parameter shapes. Defaults to False.
        log_gradients (bool): Whether to log gradient information. Defaults to False.
        log_memory (bool): Whether to log memory usage. Defaults to True.

    Returns:
        Callable: A decorator function.
    """

    def decorator(model_class: Callable) -> Callable:
        @functools.wraps(model_class)
        def wrapper(*args, **kwargs) -> nn.Module:
            model = model_class(*args, **kwargs)
            return VerboseExecution(
                model, log_params, log_gradients, log_memory
            )

        return wrapper

    return decorator


class VerboseExecution(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        log_params: bool,
        log_gradients: bool,
        log_memory: bool,
    ):
        super().__init__()
        self.model = model
        self.log_params = log_params
        self.log_gradients = log_gradients
        self.log_memory = log_memory
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            module.__name__ = name
            module.register_forward_hook(self._create_forward_hook(name))
            if self.log_gradients:
                module.register_full_backward_hook(
                    self._create_backward_hook(name)
                )

    def _create_forward_hook(self, name: str):
        def hook(
            module: nn.Module,
            input: Union[Tensor, tuple],
            output: Union[Tensor, tuple],
        ):
            log_dict: Dict[str, Any] = {
                "module": name,
                "input_shape": self._get_shape(input),
                "output_shape": self._get_shape(output),
            }

            if self.log_params:
                log_dict["parameters"] = {
                    param_name: tuple(param.shape)
                    for param_name, param in module.named_parameters()
                }

            if self.log_memory:
                log_dict["memory"] = self._get_memory_usage()

            logger.info(log_dict)

        return hook

    def _create_backward_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: Union[Tensor, tuple],
            grad_output: Union[Tensor, tuple],
        ):
            log_dict: Dict[str, Any] = {
                "module": name,
                "grad_input_shape": self._get_shape(grad_input),
                "grad_output_shape": self._get_shape(grad_output),
            }
            logger.info(f"Backward pass: {log_dict}")

        return hook

    @staticmethod
    def _get_shape(tensor_or_tuple: Union[Tensor, tuple]) -> Union[tuple, list]:
        if isinstance(tensor_or_tuple, tuple):
            return [
                VerboseExecution._get_shape(t)
                for t in tensor_or_tuple
                if t is not None
            ]
        elif isinstance(tensor_or_tuple, Tensor):
            return tuple(tensor_or_tuple.shape)
        else:
            return None

    @staticmethod
    def _get_memory_usage() -> Dict[str, float]:
        return {
            "allocated": cuda.memory_allocated() / 1024**2,
            "cached": cuda.memory_reserved() / 1024**2,
        }

    def forward(self, *args, **kwargs) -> Any:
        start_time = time.time()
        logger.info(
            f"Model input shapes: {[self._get_shape(arg) for arg in args]}"
        )

        output = self.model(*args, **kwargs)

        end_time = time.time()
        logger.info(f"Model output shape: {self._get_shape(output)}")
        logger.info(f"Forward pass time: {end_time - start_time:.4f} seconds")

        if self.log_memory:
            logger.info(f"GPU memory usage: {self._get_memory_usage()}")

        return output
