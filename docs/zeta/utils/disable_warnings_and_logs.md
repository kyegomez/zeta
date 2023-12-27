# disable_warnings_and_logs

# zeta.utils

This module provides a set of functionalities for disabling various logs and warning messages, especially useful for cleaner outputs in Python applications, reducing the amount of noise in outputs especially during debugging or while running the application in production environments.

## Class Name: CustomFilter

This class is defined within the `disable_warnings_and_logs` function. It extends the built-in `logging.Filter` class in Python and is used to filter out some unnecesary logs. The CustomFilter class is used to silence logs based on custom conditions.

The CustomFilter class has only one method `filter` which takes a record as input and checks if it fits the unwanted_logs criteria. If it does, the method returns False which excludes the record from being added to the logger.

## Method: disable_warnings_and_logs

This function uses the CustomFilter class and disable warnings coming from a variety of places. The function works to reduce the noise in logs and outputs when you are debugging or running your application. 

To disable the warnings, this function uses a collection of techniques. It uses the warnings library to disable Python related warnings. It also adjusts the logging level of specific logger objects to stop them from firing off distracting logs. A key part of this function is the use of a custom filter which allows the function to silence logs based on custom conditions.

Below, we will describe the parameters and outputs of the `disable_warnings_and_logs` function.

__Parameters:__

The `disable_warnings_and_logs` function has no parameters. 

__Outputs:__

The `disable_warnings_and_logs` function has no return statement therefore it doesn't return anything.

__Source Code:__

```python
def disable_warnings_and_logs():
    class CustomFilter(logging.Filter):
        def filter(self, record):
            unwanted_logs = [
                "Setting ds_accelerator to mps (auto detect)",
                "NOTE: Redirects are currently not supported in Windows or"
                " MacOs.",
            ]
            return not any(log in record.getMessage() for log in unwanted_logs)

    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    logging.getLogger().setLevel(logging.WARNING)

    logger = logging.getLogger()
    f = CustomFilter()
    logger.addFilter(f)

    loggers = [
        "real_accelerator",
        "torch.distributed.elastic.multiprocessing.redirects",
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
       
