from functools import wraps
from typing import Callable


def enforce_types(func: Callable) -> Callable:
    """
    A decorator to enforce type checks on the input parameters of a function based on its annotations.

    If a parameter doesn't have a type annotation, it can be of any type.

    Args:
        func (Callable): The function whose parameters are to be checked.

    Returns:
        Callable: The wrapped function with type checks.

    Examples:
        @enforce_types
        def add(a: int, b: int) -> int:
            return a + b

        add(1, 2)  # This is fine
        add('1', '2')  # This raises a TypeError
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        arg_types = func.__annotations__

        for name, value in list(zip(arg_names, args)) + list(kwargs.items()):
            if name in arg_types and not isinstance(value, arg_types[name]):
                raise TypeError(
                    f"Argument '{name}' is not of type"
                    f" '{arg_types[name].__name__}'"
                )

        return func(*args, **kwargs)

    return wrapper
