from zeta.zeta import zeta

print(zeta)


# disable warnings
import warnings

warnings.filterwarnings("ignore")

# disable tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# disable bnb warnings and others
import logging

logging.getLogger().setLevel(logging.WARNING)


class CustomFilter(logging.Filter):
    def filter(self, record):
        msg = "Created a temporary directory at"
        return msg not in record.getMessage()


logger = logging.getLogger()
f = CustomFilter()
logger.addFilter(f)


from zeta import ops
from zeta import optim
from zeta import rl
from zeta import tokenizers
from zeta import training
from zeta import utils
from zeta import models
from zeta.nn.modules.layernorm import LayerNorm
from zeta.nn.architecture.transformer import FeedForward
from zeta import nn


# nn

# models


# utils


# training

# tokenizers

# rl

# optim

# ops
