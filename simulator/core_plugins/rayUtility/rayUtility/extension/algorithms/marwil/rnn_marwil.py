"""
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

extention of RLlib's MARWIL for RNN policy based on ray 2.5.1
1. Adds duumy RNN states into SampleBatch as postprocess.
"""
from typing import Optional, Type

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override

from ray.rllib.algorithms.marwil import MARWIL

class RNNMARWIL(MARWIL):

    @classmethod
    @override(MARWIL)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            from ASRCAISim1.plugins.rayUtility.extension.algorithms.marwil.rnn_marwil_torch_policy import (
                RNNMARWILTorchPolicy,
            )

            return RNNMARWILTorchPolicy
        elif config["framework"] == "tf":
            from ASRCAISim1.plugins.rayUtility.extension.algorithms.marwil.rnn_marwil_tf_policy import (
                RNNMARWILTF1Policy,
            )

            return RNNMARWILTF1Policy
        else:
            from ASRCAISim1.plugins.rayUtility.extension.algorithms.marwil.rnn_marwil_tf_policy import RNNMARWILTF2Policy

            return RNNMARWILTF2Policy
