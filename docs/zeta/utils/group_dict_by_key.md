# group_dict_by_key

# Module Name: Zeta.Utils 

## Group dictionary keys `group_dict_by_key` based on a condition function

The `group_dict_by_key` function in `Zeta.Utils` is a utility function that facilitates grouping keys of a dictionary based on a specified condition. The condition is defined by a custom function. 

The function returns two dictionaries where one dictionary contains the keys that meet the condition and the other dictionary contains keys that do not meet the condition. This can be useful in scenarios where you would like to separate out dictionary entries based on specific conditions.

### Function Definition

The following is the definition of the `group_dict_by_key` function:

```python
def group_dict_by_key(cond, d):
    """
    Group dictionary keys based on a condition.

    Args:
        cond (function): Condition to split dictionary.
        d (dict): The dictionary to group.

    Returns:
        tuple: Two dictionaries split based on the condition.
    """
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)
```

### Arguments:

The `group_dict_by_key` function accepts the following two arguments:

| Argument | Type | Description |
| --- | --- | --- | 
| `cond` | function | A function that defines the condition based on which the dictionary keys will be split. This function should take a key as input and return a Boolean value indicating whether the key meets the condition or not. |
| `d` | dict | The dictionary that will be split into two dictionaries based on the condition provided by the `cond` function. |

### Returns:

The `group_dict_by_key` function returns two dictionaries:

1. The first dictionary contains keys that satisfy the condition specified by the `cond` function.

2. The second dictionary contains keys that do not satisfy the `cond` function.

The returned dictionaries have the same values mapped to the same keys as the original dictionary. 

### Usage Example:

#### Example 1: 

Consider having a dictionary of student marks and the goal is to group the students into those who have scored 60 and above (pass) and below 60 (fail). The `cond` function will check if the marks are greater than or equal to 60. 

```python
students_marks = {
        "John": 85,
        "Peter": 60,
        "Tracy": 72,
        "Paul": 50,
        "Angela": 67,
        "Robert": 40
}

# define the condition function to check if marks >= 60
cond = lambda marks : marks >= 60

pass_students, fail_students = group_dict_by_key(cond, students_marks)
```

The two dictionaries returned from `group_dict_by_key` would be:

```python
pass_students = {
        "John": 85,
        "Peter": 60,
        "Tracy": 72,
        "Angela": 67,
}

fail_students = {
        "Paul": 50,
        "Robert": 40
}
```

#### Example 2:

If you have a dictionary of items and their prices, and you want to separate them into items that are below or equal to $20 and items that cost more than $20:

```python
items_prices = {
    "apple": 2,
    "orange": 3,
    "mango": 1,
    "blueberry": 5,
    "grape": 10,
    "guava": 25,
    "dragon fruit": 50,
}

# define the condition function to check if price > 20
cond = lambda price : price > 20

pricey, affordable = group_dict_by_key(cond, items_prices)
```

The returned dictionaries would be:

```python
pricey = {
    "guava": 25,
    "dragon fruit": 50,
}

affordable = {
    "apple": 2,
    "orange": 3,
    "mango": 1,
    "blueberry": 5,
    "grape": 10,
}
```

