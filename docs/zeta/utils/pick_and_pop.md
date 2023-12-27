# pick_and_pop

# Module/Function Name: pick_and_pop

## Overview 

The `pick_and_pop` function is a utility function that is specifically aimed at manipulating dictionaries. It removes specified keys from a given dictionary and then returns a new dictionary that contains the removed key-value pairs. This function can be particularly useful when you need to prune a dictionary to a simpler version that contains only desired keys-value pairs.

The `pick_and_pop` function is defined in the Zeta utility module (`zeta.utils`). A dictionary in Python is an unordered collection of data in a key-value pair format. Dictionaries can have keys and values of any datatype, which makes dictionary highly valuable and versatile data structures for handling and organizing data.

## Function Definition 

```python
def pick_and_pop(keys, d):
    """
    Remove and return values from a dictionary based on provided keys.

    Args:
        keys (list): List of keys to remove from the dictionary.
        d (dict): The dictionary to pick from.

    Returns:
        dict: A dictionary with the specified keys and their values.
    """
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))
```

## Parameters and Description 

| Parameter | Type | Default |  Description |
| --- | --- | --- | --- |
| `keys` | list | N/A | List of keys from the dictionary to be removed and returned as a new dictionary. |
| `d` | dict | N/A | The original dictionary where keys are picked and popped. |

The function pick_and_pop accepts two arguments, a list of keys and a dictionary. The keys are provided in a list, and are the ones that the user wishes to remove from the dictionary. This function returns a new dictionary composed of these key-value pairs.

## Functionality and Usage 

The `pick_and_pop` function works by iterating over the list of keys and pops each key from the dictionary. The popped value is then appended to a list of values. After all the keys have been looped over, a new dictionary is created and returned by zipping together the list of keys and the list of values.

The return type of this function is a dictionary.

### Usage Example 1
```python
d = {"name": "John", "age": 30, "city": "New York"}
keys = ["name", "city"]

result = pick_and_pop(keys, d)
print(result)  # Returns: {'name': 'John', 'city': 'New York'}
```

### Usage Example 2
```python
d = {1: "apple", 2: "banana", 3: "cherry", 4: "date"}
keys = [2, 4]

result = pick_and_pop(keys, d)
print(result)  # Returns: {2: 'banana', 4: 'date'}
```

### Usage Example 3
```python
d = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
keys = ["a", "c"]

result = pick_and_pop(keys, d)
print(result)  # Returns: {'a': [1, 2, 3], 'c': [7, 8, 9]}
```

## Additional Tips 

It's important to understand that the `pick_and_pop` function directly alters the original dictionary `d` by removing the keys from it. If you want to retain the data in the original dictionary, you should create a copy of the original dictionary and pass the copy to the `pick_and_pop` function.

## References 

- Python official documentaion: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
- Python Glossary - dictionary: https://docs.python.org/3/glossary.html#term-dictionary
- Python map() function: https://docs.python.org/3/library/functions.html#map
- Python zip() function: https://docs.python.org/3/library/functions.html#zip

After understanding this function, you will have a good knowledge of manipulating dictionaries in Python. This utility function simplifies the task of extracting certain key-value pairs from a dictionary into a new dictionary, which can be very useful in data wrangling and preprocessing tasks.
