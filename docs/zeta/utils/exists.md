# exists

# Zeta Utils Documentation

## Introduction

Zeta Utils is a simple utility library that provides utilitarian functions that can be used in a variety of general programming scenarios. The utility's functions center around various common tasks such as checking if a variable is not `None`. This document provides a deep and thorough understanding of the methods of the `zeta.utils` library with ample examples of usage.

## `exists` Function

The `exists` function belongs to the `zeta.utils` library. This function performs a simple but often recurring check in programming to determine whether the passed value is not `None`. In Python, `None` represents the absence of value and often used as a default value for arguments in the function. Let's see how to use it.


### Function Definition

```python
def exists(val: any) -> bool:
    """
    Check if the value is not None.

    Args:
        val: Any type. The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None
```

### Parameters

The `exists` function takes one argument.

| Argument | Datatype | Description                                                         |
|--------------------|----------|-------------------------------------------------------------------------------------------------|
| val              | any      | The value that you want to check if it exists (is not None). | 

### Returns

| Return Type   |  Description  |
|---------------|-------------------------------|
| bool          | Returns `True` if the `val` is not `None`, else it returns `False`. | 

### Functionality

The `exists` function checks if a value is `None`. If the value is not `None` it returns `True` indicating that the value exists. In many instances in code, there is a need to check whether a variable or argument that was passed exists or not. Instead of writing the explicit condition to check this, the `exists` function can be used.

### Examples

#### Example 1

For this basic example, we are creating a variable `x` and setting it to `None`. We are then checking the value of `x` using the `exists` function. Since `x` is `None`, `exists` will return `False`.

```python
from zeta.utils import exists

x = None
print(exists(x))  # Output: False
```

#### Example 2

In this example, we are setting `x` to an integer. When we pass `x` to `exists`, it will return `True` since `x` is not `None`.

```python
from zeta.utils import exists

x = 5
print(exists(x))  # Output: True
```

#### Example 3

Here, we are setting `x` to an empty string. Even though the string is empty, it is still not `None`. Therefore, `exists` will return `True`.

```python
from zeta.utils import exists

x = ""
print(exists(x))  # Output: True
```

The `exists` function is simple, but it can be instrumental in making code cleaner and more readable.

## Other Notes

Always remember that the `exists` function simply checks if the provided value is not `None`. It doesn’t check if the value is semantically ‘empty’ like `""` or `[]` or `{}` or `0` etc.

Consider the above examples and note how to use each function effectively in your code. It is always beneficial to grasp a deeper understanding of these utility functions to ensure error-free and efficient coding.
