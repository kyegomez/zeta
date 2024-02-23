import functools

from loguru import logger
import time
import sys


# Configure loguru logger with advanced settings
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
    backtrace=True,
    diagnose=True,
    enqueue=True,
    catch=True,
)


def log_torch_op(
    log_level: str = "DEBUG",
    log_input_output: bool = True,
    add_trace: bool = True,
    log_execution_time: bool = True,
    handle_exceptions: bool = True,
):
    """
    Decorator function that logs the details of a function call, including input arguments, output result,
    and execution time. It can also handle exceptions and add stack traces to the logs.

    Args:
        log_level (str, optional): The log level to use. Defaults to "DEBUG".
        log_input_output (bool, optional): Whether to log the input arguments and output result. Defaults to True.
        add_trace (bool, optional): Whether to add stack traces to the logs when an exception occurs. Defaults to True.
        log_execution_time (bool, optional): Whether to log the execution time of the function. Defaults to True.
        handle_exceptions (bool, optional): Whether to handle exceptions and log them. Defaults to True.

    Returns:
        function: The decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if log_execution_time:
                start_time = time.time()

            # Log function call details
            if log_input_output:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.log(
                    log_level, f"Calling {func.__name__} with args: {signature}"
                )

            try:
                result = func(*args, **kwargs)
                if log_input_output:
                    logger.log(
                        log_level, f"{func.__name__} returned {result!r}"
                    )
            except Exception as e:
                if handle_exceptions:
                    if add_trace:
                        logger.exception(f"Exception in {func.__name__}: {e}")
                    else:
                        logger.log(
                            log_level, f"Exception in {func.__name__}: {e}"
                        )
                raise  # Ensure the exception is propagated
            finally:
                if log_execution_time:
                    end_time = time.time()
                    logger.log(
                        log_level,
                        (
                            f"{func.__name__} executed in"
                            f" {end_time - start_time:.4f}s"
                        ),
                    )

            return result

        return wrapper

    return decorator
