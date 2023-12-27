# disable_warnings_and_logs

# Module Name: Zeta Utilities | Function Name: disable_warnings_and_logs

## Introduction and Overview

Zeta utilities is a module focused on providing auxiliary functionalities to help in the smoother operation of your application. In the given code, we dissect the function `disable_warnings_and_logs` which is aimed at disabling varied logs and warnings that might overshadow the crucial logs or might make your logging console look messy, thereby coming in the way of debugging or understanding the flow of events.

## Function Definition

The `disable_warnings_and_logs` function is a utility function to help clean and manage the console output by muting various warnings and logs. It does not take any arguments and does not return anything.

```python
def disable_warnings_and_logs():
    """
    Disables various warnings and logs.
    """
```
This code complex doesn't take any parameters hence the table for parameters is not applicable here.

## Core Functionality and Usage Examples

The function `disable_warnings_and_logs` works by managing warnings and logs in the following manner,

1. **Disabling warnings**: The method `warnings.filterwarnings('ignore')` is run to mute all the warnings across all python packages.

2. **Disabling tensorflow logs**: By setting `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"`, we're asking Tensorflow not to display any warning logs.

3. **Disabling bnb and other various logs**: This is achieved by setting the logging level of the root logger to warning (`logging.getLogger().setLevel(logging.WARNING)`).

4. **Silencing specific logs**: By setting up a custom filter (`CustomFilter`) added to the root logger, and disabling specific loggers that may be verbose.

5. **Disabling all loggers**: The function finally disables CRITICAL level logging (`logging.disable(logging.CRITICAL)`). This means that no logs will be displayed.

Below is an example of the usage of this function:

```python
from zeta.utils import disable_warnings_and_logs

# Calling the function
disable_warnings_and_logs()
```

This code will execute the `disable_warnings_and_logs` function and all specified logs and warnings will be disabled.

Keep in mind that once executed, `disable_warnings_and_logs` mutes different logs across the operating system. This may make the debugging process more complex as some errors may not show up in the console. It is recommended you fully understand the implications and only use this function if your console gets too messy.

## Additional Information

The function can be called at the beginning of your script, once executed all the specified logs and warnings are disabled.

This function is very handy to clean up your console from unnecessary or less meaningful log statements. However, caution should be taken in using this function as it may mute some important logs which might be necessary in crucial debugging practices.

Check out more about the Python logging module [here](https://docs.python.org/3/library/logging.html), and Tensorflow logging [here](https://www.tensorflow.org/api_docs/python/tf/get_logger) to understand about the log levels and how the logs are managed in Python.

