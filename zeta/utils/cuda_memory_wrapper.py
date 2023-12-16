import torch
import functools
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def track_cuda_memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available, skip tracking memory usage")
            return func(*args, **kwargs)

        torch.cuda.synchronize()
        before_memory = torch.cuda.memory_allocated()

        try:
            result = func(*args, **kwargs)
        except Exception as error:
            logging.error(f"Error occurs when running {func.__name__}: {error}")
            raise

        finally:
            torch.cuda.synchronize()
            after_memory = torch.cuda.memory_allocated()
            memory_diff = after_memory - before_memory
            logging.info(
                f"Memory usage of {func.__name__}: {memory_diff} bytes"
            )

        return result


return wrapper
