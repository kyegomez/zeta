# print_main

# Module Name: zeta.utils.print_main

## Function Definition

class zeta.utils.print_main(msg):
```python
Prints a message only on the main process.   

Parameters:
- msg (str): The message to be printed.
```

## Functionality & Purpose

This function serves to print messages selectively on the main process in a distributed setting. Distributed settings often clone multiple processes across different CPU cores or different machines. This means that each of these processes will have a predefined rank, where the main (or master) process usually has the rank 0. 

When dealing with distributed settings, it's quite common to observe duplicate console output from each process, which can clutter the console and make interpretability harder. This function helps to mitigate that problem by enabling messaging only from the main process, thus maintaining a clean and streamlined console output.

## Usage and Examples:

### Importing the Necessary Libraries
This function would typically be used within a project that utilises PyTorch's distributed utilities for parallel and distributed computation. So let's begin with the necessary imports:
```python
from torch import distributed as dist
import zeta.utils
```

### Example 1: Printing without Distributed Setting
   In an environment where distributed computing is not being used or available, messages will be printed normally.
```python
zeta.utils.print_main("Hello World!")
```
Console Output:
```
Hello World!
```

### Example 2: Printing with Distributed Setting
   In a distributed computing environment, the message will print only from the main process:
   
```python
# Assuming we are in a distributed environment with several processes running this code
if dist.is_available():
    zeta.utils.print_main("Hello from main process!")
```
Console Output:
```
# Note: This message will only be printed once, since only the main process (rank 0) gets to execute the print function.
Hello from main process!
```

Remember that in this scenario, if the current process is not the main process (i.e., its rank is not 0), the function simply won't do anything. This is beneficial to avoid repetitively printing the same message in a distributed setting. 

Remember to ensure your distributed environment is properly initialized before using distributed functionalities.
   
### Example 3: Handling both Non-Distributed and Distributed Settings
   This function is designed to handle both non-distributed and distributed settings, as shown below:
   
```python
# main function
def main():
    # distributing tasks between processes.
    print_main("This message is from main process only.")   

if __name__ == "__main__":
    main()
```

Here, `dist.is_available()` checks if distributed processing is available. If so, it verifies if the rank is 0 (i.e., checks if the process is the main one). If both conditions are true, it goes ahead and prints the message. If distributed processing isn't available, it directly prints the message, effectively handling both scenarios.
