# string_begins_with

# Module Name: **zeta.utils**

## Introduction

The `zeta.utils` module is a handy utilities toolkit for Python, which includes a variety of useful functions for data processing and manipulation. A noteworthy function in this module is `string_begins_with`. It provides a quick and easy way to check if a string starts with a particular prefix. Though it seems a simple function, it is essential in many data preprocessing tasks such as checking the file paths, URLs, filenames, and prefix-based conditional data manipulation.

## Functionality Overview

The `string_begins_with` function takes two arguments: `prefix` and `str`. It checks if the given string `str` commences with the specified `prefix` and returns a boolean value accordingly.

Now, let's explore the function syntax, parameters, and usage.

## Function Definition and Parameters

The `string_begins_with` is defined as follows:

```Python
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

Here's a breakdown of its parameters:

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `prefix` | str  | The prefix that we need to check for at the start of the string. |
| `str`    | str  | The string that we need to inspect. |

## Functionality and Usage

The primary usage of the `string_begins_with` function is to check if a string begins with a specific prefix. In Python, we have the `str.startswith()` function that performs this check. The `string_begins_with` function is essentially a wrapper around this built-in function providing a clear and expressive syntax.

The function `string_begins_with` is a pure function in that it neither modifies the actual inputs nor does it rely on or alter any external state. It only produces the result based on the given inputs.

Here are a few usage instances:

**Example 1** - Basic usage:
```Python
from zeta.utils import string_begins_with

print(string_begins_with('data', 'database')) # Output: True
print(string_begins_with('data', 'base')) # Output: False
```

**Example 2** - Handling case-sensitivity:
```Python
from zeta.utils import string_begins_with

print(string_begins_with('Data', 'database')) # Output: False
print(string_begins_with('Data', 'Database'))  # Output: True
```

**Example 3** - Using with list comprehension for data preprocessing:
```Python
from zeta.utils import string_begins_with

data = ['apple', 'android', 'blackberry', 'windows', 'android_tv']
android_data = [item for item in data if string_begins_with('android', item)]

print(android_data) # Output: ['android', 'android_tv']
```

Cognizant of Python's inbuilt `startswith` function, `string_begins_with` complements it by providing a more meaningful syntax that enhances the code readability, especially for those new to Python programming. Through this documentation, we hope you'll be able to integrate `string_begins_with` into your code and simplify your string prefix checks. Happy Programming!
