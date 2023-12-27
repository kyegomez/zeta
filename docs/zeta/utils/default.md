# default

# Zeta.Utils - Python Documentation

## Table of Contents
1. [Overview](#overview)
2. [Code Documentation](#codedocumentation)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Additional Information](#additionalinfo)
6. [References and Other Resources](#references)

---

<a name="overview"/>

# 1. Overview

`Zeta.Utils` is a Python module that contains auxiliary functions to ease and manage general programming tasks. The module is built to operate smoothly with Python and its ecosystem. This document has been created to guide users in the proper use of the library, especially in using the `default` function present in `Zeta.Utils`.

This documentation will provide a comprehensive insight into the purpose, functionality, usage, and worked out examples of the `default` function. The document is explicitly made in a step-by-step manner to provide exhaustive information on how to use the function effectively along with various scenarios and cases.

---

<a name="codedocumentation"/>

# 2. Code Documentation

### Function Name: default

```python
def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        val (Any): The value to check.
        d (Any): The default value to return if val is None.

    Returns:
        Any: The value if it exists, otherwise the default value.
    """
    return val if exists(val) else d
```

**Parameters:**

| Parameter | Data Type | Default Value | Description |
| --- | --- | --- | --- |
| val | Any | - | The value to check |
| d | Any | - | The default value to return if val is None |

**Returns:**

The return value is of type `Any` and is the value of `val` if it exists, else it's the default value `d`.

---

<a name="usage"/>

# 3. Usage

The `default` function in `Zeta.Utils` is a utility function primarily used to provide a "default" return value in case the checked value is None.

To use the `default` function, import the function into your Python script and call the function with two arguments, the value to check if it exists (`val`), and the default value to return if the value does not exist (`d`). 

The function will then return the existing `val` if it is not None, otherwise, it will return the default value `d`.

---

<a name="examples"/>

# 4. Examples

Below are example cases, demonstrating how the `default()` function can be used in a Python script.

**Example 1**

Provides a simple example showing the use of `default()`:

```python
from zeta.utils import default

result = default(None, "Default Value")
print(result)  # Output: Default Value
```

In the above code, the `default` function is called with `None` as the `val` and "Default Value" as `d`. Since `val` is `None`, the function returns `d` which is "Default Value".

**Example 2**

Provides an example where `val` is not None:

```python
from zeta.utils import default

data = "Test Value"
result = default(data, "Default Value")
print(result)  # Output: Test Value
```

Above, the `default` function is called with "Test Value" as `val` and "Default Value" as `d`. Since `val` is not `None`, the function returns `val` which is "Test Value".

**Example 3**

Shows use of `default` with data structures:

```python
from zeta.utils import default

data = []
default_value = [1, 2, 3]
result = default(data, default_value)
print(result)  # Output: []
```

In this example, even if `data` is an empty list, it's not `None`, so the `default` function returns `data` as the output.

---

<a name="additionalinfo"/>

# 5. Additional Information

The function `default` is a versatile utility for handling `None` scenarios. However, it may mask issues wherein `None` is an unexpected value. Developers are advised to use `default` along with proper error handling or assertions to ensure that `None` values are detected and handled when not expected.

In scenarios where a false-y value like `0, "", [], or {}` should be replaced with a default, it's recommended to use the standard or in Python like `val or d`.

<a name="references"/>

# 6. References and Other Resources

For more details on Python, consult the Python documentation at [docs.python.org](https://docs.python.org/).

Further information on Zeta.Utils and the `default`
