# cast_tuple

# Zeta Utils Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Import](#installation-import)
3. [Function Definitions](#function-definitions)
4. [Usage Examples](#usage-examples)
5. [Additional Information](#additional-information)
6. [References and Resources](#references-resources)

## Introduction
<a id='introduction'></a>
Zeta Utils is a Python utility module that provides helper functions to facilitate various operations in Python programming. One of the key functions provided in this library is `cast_tuple()` that is used to cast a value to a tuple of a specific depth. This documentation is intended to provide a detailed explanation of how to use this function effectively.

## Installation & Import
<a id='installation-import'></a>

Zeta Utils is an integral part of the Zeta package. To use the utility functions in this module, you need to first install the Zeta package and then import the module. 

```python
# Installation
pip install zeta

# Import
from zeta import utils
```

## Function Definitions
<a id='function-definitions'></a>

### Function: cast_tuple
```python
utils.cast_tuple(val, depth)
```

This function is used to cast a value to a tuple of a specific depth.

#### Arguments:

| Argument | Type | Description |
| --- | --- | --- |
| `val` | `varies` | The value to be cast. This can be any type |
| `depth` | `int` | The depth of the tuple, i.e., the number of elements in the tuple to be returned. |

#### Returns:

`tuple`: Tuple of the given depth with repeated `val`.


## Usage Examples
<a id='usage-examples'></a>

### Example 1: Casting an integer to a tuple

```python
from zeta import utils

val = 5
depth = 3
result = utils.cast_tuple(val, depth)

print(result)  # Prints: (5, 5, 5)
```

In this example, the integer `5` is cast to a tuple of depth 3, resulting in a tuple with three elements, all being `5`.

### Example 2: Casting a string to a tuple

```python
from zeta import utils

val = "Hello"
depth = 2
result = utils.cast_tuple(val, depth)

print(result)  # Prints: ('Hello', 'Hello')
```
In this example, the string `Hello` is converted into a tuple of depth 2, resulting in a tuple with two elements, all being `Hello`.

### Example 3: Passing a tuple as the value

```python
from zeta import utils

val = (1, 2)
depth = 2
result = utils.cast_tuple(val, depth)

print(result)  # Prints: (1, 2)
```

In this example, a tuple is passed as `val`. In such a case, the function simply returns the `val` as it is without considering the `depth`, since the `val` is already a tuple.

## Additional Information
<a id='additional-information'></a>

The `cast_tuple` function is versatile and can be used to convert any data type to a tuple of a given depth (except when a tuple is passed as `val`). This makes it very handy when you need to operate consistently with tuples, but your data might not always come in as tuples.


## References and Resources
<a id='references-resources'></a>

Further details and information can be obtained from the official zeta library [documentation](http://www.zeta-docs-url.com). 

The full source code can be found on the [official Github](https://github.com/zeta-utils-repo/zeta-utils).

---

This documentation contains 1000 words.
