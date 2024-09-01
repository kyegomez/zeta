# group_by_key_prefix

# Module/Function Name: group_by_key_prefix

## Overview
This utility function group_by_key_prefix contained in the zeta.utils library, serves to provide functionality that allows users to easily group items in a dictionary based on the prefix of keys. This is particularly useful when handling complex nested dictionaries where classifying and grouping keys can enhance readability and processing.

We see this functionality in many practical scenarios such as parsing and grouping HTTP headers, processing JSON data, or categorizing data in large datasets - all based on prefixed keys.

## Function Definition

### `group_by_key_prefix(prefix, d)`

#### Parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| prefix | str | This is the prefix that the function checks for in each key of the passed dictionary | - |
| d | dict | This is the dictionary that needs to be processed and grouped | - |
 
The function takes two parameters: `prefix` which is a string and `d` which is a dictionary. 

The function checks each key of the passed dictionary `d` and groups them based on whether they start with the specified `prefix` or not. 

#### Returns:
The function returns a tuple of two dictionaries. One dictionary contains all items where keys start with the given prefix and the other dictionary contains all items where keys do not start with the given prefix.

```python
def group_by_key_prefix(prefix, d):
    """
    Group dictionary items by keys that start with a specific prefix.

    Args:
    prefix (str): The prefix to check for.
    d (dict): The dictionary to group.

    Returns:
    tuple: Two dictionaries split based on the prefix condition.
    """
    return group_dict_by_key(partial(string_begins_with, prefix), d)
```

## Function Usage & Examples

Let's go through examples that illustrate the usage of this function:

### Example 1 - Basic Scenario:

In a scenario where we have a dictionary of various fruits and we wish to group them based on the first letter of the fruit's name. For example, we can choose "a" as our prefix. Here's how we can process the dictionary:

```python
import zeta.utils as zutils

fruits = {
    "apple": 5,
    "avocado": 2,
    "banana": 4,
    "blackberry": 3,
    "cherry": 7,
    "apricot": 1,
}

prefix = "a"
grouped_fruits = zutils.group_by_key_prefix(prefix, fruits)
print(grouped_fruits)
```

### Example 2 - Empty Dictionary:

In the scenario where we pass an empty dictionary, we will receive two empty dictionaries in return as there are no keys to process:

```python
import zeta.utils as zutils

empty_dict = {}

prefix = "a"
grouped_dict = zutils.group_by_key_prefix(prefix, empty_dict)
print(grouped_dict)  # output: ({}, {})
```

### Example 3 - No Keys With Specified Prefix:

If there are no keys in the dictionary that start with the specified prefix, then one of the dictionaries returned in the tuple will be empty:

```python
import zeta.utils as zutils

fruits = {"banana": 4, "blackberry": 3, "cherry": 7}

prefix = "a"
grouped_fruits = zutils.group_by_key_prefix(prefix, fruits)
print(grouped_fruits)  # output: ({}, {'banana': 4, 'blackberry': 3, 'cherry': 7})
```

## Additional Tips & Best Practices:
1. Prefix search is case-sensitive. If keys contain capital letters, make sure to provide a capital letter as the prefix too if you're looking for an exact match.
2. This function does not search prefixes recursively. If dictionary values are themselves dictionaries, the function will not process keys for those nested dictionaries.
3. Be mindful of dictionary key types. This function will not work if keys are not string type.

## References & Further Reading:
1. Python Dictionary Official Documentation: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
2. Functional Programming in Python: https://docs.python.org/3/howto/functional.html

This documentation provides an explanation on using the `group_by_key_prefix` utility function. For details on other functions provided by the `zeta.utils` library, refer to the respective documentation.
