# once

# Zeta Utils Library Documentation

## Contents

1. [Overview](#overview)
2. [Detailed Function Documentation](#Detailed-Function-Documentation)
   - [once](#once)
3. [Usage Guides](#Usage-Guides)

## <a name="overview"></a> Overview

Zeta utils library, in this case, contains a single function `once`, a decorator which ensures that the function it wraps is only called once. This utility function can be extremely useful in situations where duplicate function calls could lead to unnecessary redundancy or inefficiencies.

## <a name="Detailed-Function-Documentation"></a> Detailed Function Documentation

### <a name="once"></a> once

#### Signature

```python
@once
def FUNCTION_NAME(ARGS)
```

#### Description

A decorator function that ensures the function it wraps is only called once. This prevents duplicate function calls, thereby improving efficiency in situations where duplicate function calls could be redundant or detrimental to the performance of your program.

#### Parameters

| Name | Type     | Description   | 
|------|----------|---------------|
| fn   | function | The function to be wrapped and executed only once.| 

#### Returns

The wrapped function that will run only once.


#### Source code

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
    def inner(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            return fn(*args, **kwargs)

    return inner
```

## <a name="Usage-Guides"></a> Usage Guides

### Example 1: Basic Usage

In this example, we will create a simple function that returns a greeting. We will use the `once` decorator to ensure the function only prints the greeting once, even if the function is called multiple times.

```python
from functools import wraps
# Include your once function in here.

def once(fn):
    called = False

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            return fn(*args, **kwargs)

    return inner

@once
def greet(name):
    return f"Hello {name
