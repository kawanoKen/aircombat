"""modification of RLlib's MARWIL for recurrent policies based on ray 2.5.1
1. Adds recurrent states into SampleBatch as postprocess.
2. Fixes memory leak in torch version reported in the issue #21291 but not merged as of ray 1.13.
"""
from .rnn_marwil import (
    RNNMARWIL,
)
from .rnn_marwil_tf_policy import (
    RNNMARWILTF1Policy,
    RNNMARWILTF2Policy,
)
from .rnn_marwil_torch_policy import RNNMARWILTorchPolicy

__all__ = [
    "RNNMARWIL",
    "RNNMARWILTF1Policy",
    "RNNMARWILTF2Policy",
    "RNNMARWILTorchPolicy",
]
