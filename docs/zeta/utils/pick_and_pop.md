# pick_and_pop

# Documentation for `pick_and_pop` function in `zeta.utils` 

## Introduction

The `pick_and_pop` function in the `zeta.utils` library is a handy utility function for dictionary manipulation. It provides an efficient way to extract specific key-value pairs from a Python dictionary and also simultaneously remove these key-value pairs from the original dictionary. This operation is beneficial when needing a subset of data from a large dictionary for further processing while removing it from the parent dictionary for memory efficiency.

## Class or Function Definition

Function signature:

```python
pick_and_pop(keys: list, d: dict) -> dict
```

## Parameters

The `pick_and_pop` function takes two parameters. 

|Parameter|Type|Description|
|---------|----|-----------|
|`keys`|list|List of keys to remove from the dictionary|
|`d`|dict|The dictionary to pick from|

## Returns

The `pick_and_pop` function returns a new dictionary containing the key value pairs specified in the `keys` list parameter.

## Functionality and Usage

The `pick_and_pop` function makes use of the `pop` method native to Python dictionaries. The `pop` method is specified in a lambda function which is then mapped onto the list of `keys`. This effectively extracts the value associated to each key in `keys` from dictionary `d` and also removes this key-value pair from `d`.

A new dictionary, containing the key-value pairs specified in `keys`, is then created and returned using the built-in `dict` function in combination with the `zip` function to pair each key in `keys` with its corresponding value.

## Usage Examples

### Example 1: Basic Usage

```python
# import the function
from zeta.utils import pick_and_pop

# initialize a dictionary
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
print('Original d:', d)

# specify the keys we want to pop from the dictionary
keys = ['a', 'c']

# apply the function
res = pick_and_pop(keys, d)
print('Result:', res)
print('Modified d:', d)

# Output:
# Original d: {'a': 1, 'b': 2, 'c': 3, 'd': 4}
# Result: {'a': 1, 'c': 3}
# Modified
