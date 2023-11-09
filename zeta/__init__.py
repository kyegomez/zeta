import logging
import os
import warnings

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

from zeta.nn import *
from zeta import models
from zeta import utils
from zeta import training
from zeta import tokenizers
from zeta import rl
from zeta import optim
from zeta import ops
