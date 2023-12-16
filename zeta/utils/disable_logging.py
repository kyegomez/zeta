import logging
import os
import warnings


def disable_warnings_and_logs():
    """Disable warnings and logs.

    Returns:
        _type_: _description_
    """
    # disable warnings
    warnings.filterwarnings("ignore")

    # disable tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # disable bnb warnings and others
    logging.getLogger().setLevel(logging.WARNING)

    class CustomFilter(logging.Filter):
        def filter(self, record):
            msg = "Created a temporary directory at"
            return msg not in record.getMessage()

    logger = logging.getLogger()
    f = CustomFilter()
    logger.addFilter(f)


disable_warnings_and_logs()
