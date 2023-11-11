from zeta.ops.main import *
from zeta.ops.softmax import *
from zeta.ops.unitwise_norm import unitwise_norm
from zeta.ops.mos import MixtureOfSoftmaxes

from zeta.ops.softmax import (
    standard_softmax,
    # selu softmax,
    selu_softmax,
    # 2. Sparsemax,
    sparsemax,
    # 3. Local Softmax,
    local_softmax,
    # 4. Fast Softmax,
    fast_softmax,
    # 5. Sparse Softmax,
    sparse_softmax,
    # 6. gumbelmax,
    gumbelmax,
    # 7. Softmax with temp,
    temp_softmax,
    # 8. logit scaled softmax,
    logit_scaled_softmax,
    # 9. norm exponential softmax,
    norm_exp_softmax,
)

__all__ = [
    "standard_softmax",
    # selu softmax,
    "selu_softmax",
    # 2. Sparsemax,
    "sparsemax",
    # 3. Local Softmax,
    "local_softmax",
    # 4. Fast Softmax,
    "fast_softmax",
    # 5. Sparse Softmax,
    "sparse_softmax",
    # 6. gumbelmax,
    "gumbelmax",
    # 7. Softmax with temp,
    "temp_softmax",
    # 8. logit scaled softmax,
    "logit_scaled_softmax",
    # 9. norm exponential softmax,
    "norm_exp_softmax",
    "unitwise_norm",
]
