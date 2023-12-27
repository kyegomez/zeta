# string_begins_with

# Module/Function Name: string_begins_with

```python
def string_begins_with(prefix, str):
    """
    Check if a string begins with a specific prefix.

    Args:
        prefix (str): The prefix to check for.
        str (str): The string to check.

    Returns:
        bool: True if string starts with prefix, False otherwise.
    """
    return str.startswith(prefix)
```
## 1: Introduction

The `string_begins_with` function is a simple utility function that checks whether a given string begins with a specified prefix. It is part of the `zeta.utils` library and represents a common application in string manipulation.

## 2: Parameters

The function accepts the following arguments as required:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| prefix | str  | The prefix to check for. |
| str     | str  | The string to check. |

## 3: Output

The function returns a boolean value:

| Value | Type | Description |
| ----- | ---- | ----------- |
| output | bool | True if string starts with prefix, False otherwise. |

## 4: Functionality and Usage

The `string_begins_with` function is quite straightforward. It leverages Python's built-in `str.startswith` method to determine if the string `str` starts with the provided `prefix`. If so, the function returns `True`; otherwise, it returns `False`.

You can use the `string_begins_with` function in any situation where you need to check whether a given string starts with a specific substring. This can be especially useful in text processing or data cleaning tasks, where you might need to categorize or filter strings based on their prefixes.

Here are three examples showing how to use the `string_begins_with` function:

**Example 1 Basic usage**

```python
from zeta.utils import string_begins_with

str = "Hello, world"
prefix = "Hello"
result = string_begins_with(prefix, str)
print(result) # Output: True
```

**Example 2 When string does not start with prefix**

```python
from zeta.utils import string_begins_with

str = "Hello, world"
prefix = "Hi"
result = string_begins_with(prefix, str)
print(result) # Output: False
```

**Example 3 With a numeric prefix**

```python
from zeta.utils import string
