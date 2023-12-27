# exists

# Module/Function Name: exists

Python module `zeta.utils` contains a function named `exists`. This utility function quickly checks if a given variable or value is not `None` and returns a boolean value of `True` if it not None and `False` otherwise.

It is a simple yet powerful utility function that has numerous use cases in programming and data processing where checking the existence of a particular value is mandatory.

## Definition

```python
def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None
```

## Parameters

**val**: It's the only parameter function accepts of any data type including `None`. It is the value for which you want to perform the existence check.

## Return

The function returns a boolean value - either `True` or `False`.

Returns `True` when the passed value is not None, and `False` when the value is None.

## Usage

The `exists` function is incredibly simple to use:

1. Import the function from the `zeta.utils` module.
2. Pass the value (the existence of which you want to check) to the function.
3. The function will return a boolean value based on the existence of the passed value.

## Code example:

```python
from zeta.utils import exists

x = "Hello, world!"
z = None

print(exists(x))  # prints: True
print(exists(z))  # prints: False
```

In the above example, the `exists` function returns `True` for the variable `x` as it is not `None`. 

It then returns `False` for the variable `z` as its value is indeed `None`.

## Practical application scenarios

**Case 1:**
When processing incoming data, you want to check if a certain piece of data exists before performing operations on it.

```python
from zeta.utils import exists

data = get_incoming_data()

if exists(data):
    process_data(data)
else:
    print("No data to process")
```

**Case 2:**
Ensuring a function argument is not None before performing an operation.

```python
from zeta.utils import exists

def some_operation(a, b, c):
    if exists(c):
        return
