# print_main

# Zeta Utils Library - print_main function documentation

## Overview
Welcome to the documentation of the `print_main` function provided in the `zeta.utils` library. This function serves a purpose in a distributed data setup where multiple processes are running concurrently. Often in such setups, avoiding duplication of logs or messages is desirable, and this function helps to achieve it by ensuring that specific messages get printed only on the main process.

This utility function can be incredibly useful when debugging or logging information in a distributed setting, providing cleaner logs and easier debugging. This documentation will guide you on how to use the `print_main` function, detailing its arguments, usages, and examples.

## Function Definition

```python
def print_main(msg):
    """Print the message only on the main process.

    Args:
        msg (_type_): _description_
    """
    if dist.is_available():
        if dist.get_rank() == 0:
            print(msg)
    else:
        print(msg)
```

## Arguments
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `msg` | string | The message that should be printed by the main process |


The `print_main` function accepts a single argument:

- `msg`: (string) This is the message to be printed to the console. The message should be of the type `string`.

## Usage

The `print_main` function is quite straightforward to use. Here, we detail how to use this function in three different ways:

### 1. Basic Functionality

This is the simplest and most basic example demonstrating the usage of the `print_main` function.

```python
import torch.distributed as dist
from zeta.utils import print_main

# Within your main function
print_main("This is a test message.")
```

### 2. Testing with Various Messages

In the following example, we tweak the earlier sample code and add a loop to send different messages. In a real-life implementation, you would replace this with your application-specific messages.

```python
import torch.distributed as dist
from zeta.utils import print_main

# Within your main function
for i in range(5):
    print_main(f"This is test message number: {i}")
```

### 3. Using the Function in a Multithreaded Environment

Assume you have a multithreaded setup where multiple processes are running concurrently, and you want to print some
