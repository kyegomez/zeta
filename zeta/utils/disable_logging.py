import logging
import os
import warnings

def disable_warnings_and_logs():
    """
    Disables various warnings and logs.
    """
    class CustomFilter(logging.Filter):
        def filter(self, record):
            unwanted_logs = [
                "Setting ds_accelerator to mps (auto detect)",
                (
                    "NOTE: Redirects are currently not supported in Windows or"
                    " MacOs."
                ),
            ]
            return not any(log in record.getMessage() for log in unwanted_logs)

    # disable warnings
    warnings.filterwarnings("ignore")

    # disable tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # disable bnb warnings and others
    logging.getLogger().setLevel(logging.WARNING)

    # add custom filter to root logger
    logger = logging.getLogger()
    f = CustomFilter()
    logger.addFilter(f)

    # disable specific loggers
    loggers = [
        "real_accelerator",
        "torch.distributed.elastic.multiprocessing.redirects",
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)

    # disable all loggers
    logging.disable(logging.CRITICAL)