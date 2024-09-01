# once

# Function Name: once

## Overview and Introduction

In a variety of contexts, whether while initializing some variables, setting up logging, or ensuring some heavy computation isn't undertaken multiple times, there are scenarios where you might want to ensure a function is executed only once. The `once` function is a Python decorator that took up this challenge. By using it, we guarantee a wrapped function is called only for the first time it is invoked.

The `once` function meets this requirement by retaining a flag `called` in its closure. This flag tracks whether or not a function has been called before. When the function is called, it checks the flag. If the flag is false (`False`), implying the function hasn't been called before, it allows the function to execute and toggles the flag. If the flag is true (`True`), indicating the function has been called before, it simply returns, preventing the function execution.

## Function Definition

Let's consider the structure and details of the `once` function. It accepts a single argument, `fn`, which is the function to be wrapped. The function is returned as the output after being wrapped in a closure that maintains the `called` flag. 

```python
def once(fn):
    """
    Decorator to ensure the function is only called once.

    Args:
       fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner
```

| Argument | Type | Description |
| --- | --- | --- |
| fn | function | The function to wrap. | 

## Functionality and Usage 

The `once` function ensures that the annotated function `fn` is executed only once - the first time it's called. For all subsequent calls, it immediately returns without executing the function `fn`. The `once` decorator therefore is particularly useful in scenarios where a specific function should not or need not be executed more than once. 

### Example - Initial Setup Function

Let's demonstrate the `once` function with a setup function, `setup()`. This could represent any kind of initialization logic that should only be run once:

```python
@once
def setup():
    print("Setting up...")


# The setup() function is invoked twice.
setup()  # Prints: 'Setting up...'
setup()  # Doesn't print anything.
```

### Example - Heavy Computation Function

Here is an example where a computation should only be executed once:

```python
@once
def heavy_computation():
    print("Doing heavy computation...")
    # long running computation


# The heavy_computation() function is invoked twice.
heavy_computation()  # Prints: 'Doing heavy computation...'
heavy_computation()  # Doesn't print anything.
```

### Example - State Initialisation 

If you are dealing with a stateful class and need to initialize something only once, `once` decorator can come handy:

```python
class MyClass:
    @once
    def initialize(self):
        print("Initializing state...")


# MyClass object is created, the initialize function is called twice.
obj = MyClass()
obj.initialize()  # Prints: 'Initializing state...'
obj.initialize()  # Doesn't print anything.
```

In each of the above examples, similarly, the decorated function `setup()`, `heavy_computation()` and `initialize()` were called multiple times but executed only once.

The use of `once` decorator provides a convenient way to ensure specific functions only run their core execution once, while allowing them to be flexibly called without caution multiple times elsewhere in code or scripts. This helps maintain cleaner and more predictable code especially when dealing with initializations and one-time setups.
