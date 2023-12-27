# group_dict_by_key

# Module/Function Name: group_dict_by_key (Internally within `zeta.utils`)

Function `group_dict_by_key` is a utility function which is designed to split specific dictionary based on the condition provided by the user. This function accepts two arguments: a condition (a function), and a dictionary. The key feature of this function is the implicit usage of the user-defined function to be used as a condition to split the dictionary on. This function allows users to take a very flexible approach in handling, processing, and manipulating dictionary objects in Python.

## Function Signature

```python
def group_dict_by_key(cond: function, d: dict) -> Tuple[dict, dict]
```

This function takes in a `function` parameter which will be used to divide the dictionary into two parts, and the `dictionary` to be divided. The function can be named according to the condition of use, and its definition is entirely up to the user. The dictionary `d` is the dictionary to be divided.

## Function Parameters

| Parameter | Type | Description | Default Value |
| ------- | -------- | ------------------------------------------------------ | ---------------- |
| cond | function | User-defined function to be used to split the dictionary | NA |
| d | dict | Dictionary to be divided | NA |

## Returns

This function returns a `Tuple[dict, dict]`. Specifically, it outputs a tuple of dictionaries divided based on the condition provided.

## How it Works

The function `group_dict_by_key` starts by initializing two empty dictionaries `return_val`. It then iterates through every key in the input dictionary `d`. For each key, it evaluates the user-defined condition function `cond(key)`. If the condition is matched, the current key and value pair is added to the first new dictionary. If the condition is not matched, the current element is added to the second new dictionary. Therefore, the function iterates through all key-value pairs in the input dictionary and divide them into two dictionaries based on whether or not they meet the user-defined condition.

## Examples and Usage

#### Import

In order to use this function, you must first understand how to import it. Here is an example of how you might do this:

```python
from zeta.utils import group_dict_by_key
```

#### Use

Here are three different examples of how you'd use `group_dict_by_key` function:

1. Grouping dictionary keys based on length: 

```python
cond =
