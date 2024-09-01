import logging
import time

import torch
import torch.fx
import torch.jit
from torch import nn
from torch.quantization import quantize_dynamic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hyper_optimize(
    torch_fx=True,
    torch_script=True,
    torch_compile=True,
    quantize=False,
    mixed_precision=False,
    enable_metrics=False,
):
    """
    Decorator for PyTorch model optimizations including JIT, FX, Compile, Quantization, and Mixed Precision.

    Args:
        torch_fx (bool): Flag indicating whether to apply torch.fx transformation. Default is True.
        torch_script (bool): Flag indicating whether to apply torch.jit script. Default is True.
        torch_compile (bool): Flag indicating whether to apply torch.compile. Default is True.
        quantize (bool): Flag indicating whether to apply model quantization. Default is False.
        mixed_precision (bool): Flag indicating whether to use mixed precision. Default is False.
        enable_metrics (bool): Flag indicating whether to enable performance metrics. Default is False.

    Returns:
        decorator (function): Decorator function that applies the specified optimizations to the target function.

    Example::
    @hyper_optimize(
        torch_fx=False,
        torch_script=False,
        torch_compile=True,
        quantize=True,
        mixed_precision=True,
        enable_metrics=True,
    )
    def model(x):
        return x @ x

    out = model(torch.randn(1, 3, 32, 32))
    print(out)

    """

    def decorator(fn):
        if isinstance(fn, nn.Module):
            target = fn.forward
        else:
            target = fn

        # Apply torch.fx transformation
        if torch_fx:
            try:
                fx_transformed = torch.fx.symbolic_trace(fn)
                target = fx_transformed
            except Exception as e:
                logger.warning("torch.fx transformation failed: %s", e)

        # Apply torch.jit script
        if torch_script:
            try:
                jit_scripted = torch.jit.script(target)
                target = jit_scripted
            except Exception as e:
                logger.warning("torch.jit scripting failed: %s", e)

        # Apply torch.compile
        if torch_compile and hasattr(torch, "compile"):
            try:
                compiled_fn = torch.compile(target)
                target = compiled_fn
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

        # Apply Quantization
        if quantize:
            try:
                target = quantize_dynamic(target)
            except Exception as e:
                logger.warning("Model quantization failed: %s", e)

        # Wrapper for mixed precision
        def mixed_precision_wrapper(*args, **kwargs):
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                return target(*args, **kwargs)

        # Performance Metrics
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = mixed_precision_wrapper(*args, **kwargs)
            end_time = time.time()
            logger.info("Execution time: %f seconds", end_time - start_time)
            return result

        return (
            wrapper
            if enable_metrics
            else (mixed_precision_wrapper if mixed_precision else target)
        )

    return decorator
