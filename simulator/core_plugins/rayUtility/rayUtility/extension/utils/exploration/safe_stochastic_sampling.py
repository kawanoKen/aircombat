# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""ray RLlibで探索を行うためのクラスであるRandomとStochasticSamplingがnestedな行動空間に対応していないため、
    TupleとDictを含むnestedな行動空間でも使用できるように一部改変したもの
"""
import numpy as np
import gymnasium as gym
import tree  # pip install dm_tree

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import ModelV2
from ray.rllib.utils import force_tuple
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.utils.framework import get_variable
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class SafeRandom(Random):
    """TupleとDictを含むnestedな行動空間でもランダムサンプリングを使用できるように修正したクラス
    """
    
    def get_torch_exploration_action(
        self, action_dist: ActionDistribution, explore: bool
    ):
        if explore:
            req = force_tuple(
                action_dist.required_model_output_shape(
                    self.action_space, getattr(self.model, "model_config", None)
                )
            )

        # TupleとDictは中身を再帰で辿る
        # Boxと(Multi)Discreteに
            def add_batch_dim_and_to_torch(s):
                if isinstance(s, gym.spaces.Tuple):
                    return tuple(add_batch_dim_and_to_torch(v) for v in s.spaces)
                elif isinstance(s, gym.spaces.Dict):
                    return {k: add_batch_dim_and_to_torch(v) for k, v in s.spaces.items()}
                else:
                    # Add a batch dimension?
                    if len(action_dist.inputs.shape) == len(req) + 1:
                        batch_size = action_dist.inputs.shape[0]
                        a = np.stack([s.sample() for _ in range(batch_size)])
                    else:
                        a = s.sample()
                    a = torch.from_numpy(a).to(self.device)
                    return a
            action = add_batch_dim_and_to_torch(self.action_space)
        else:
            action = action_dist.deterministic_sample()

        # logpはtf_utils.pyのzero_logps_from_actionsと同じ方法で計算
        action_component = tree.flatten(action)[0]
        logp = torch.zeros_like(action_component, dtype=torch.float32)
        while len(logp.shape) > 1:
            logp = logp[:, 0]

        return action, logp


class SafeStochasticSampling(StochasticSampling):
    """TupleとDictを含むnestedな行動空間でもランダムサンプリングを使用できるように修正したクラス
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        *,
        framework: str,
        model: ModelV2,
        random_timesteps: int = 0,
        **kwargs
    ):
        """Initializes a StochasticSampling Exploration object.

        Args:
            action_space: The gym action space used by the environment.
            framework: One of None, "tf", "torch".
            model: The ModelV2 used by the owning Policy.
            random_timesteps: The number of timesteps for which to act
                completely randomly. Only after this number of timesteps,
                actual samples will be drawn to get exploration actions.
        """
        assert framework is not None
        super().__init__(action_space, model=model, framework=framework, **kwargs)

        # Create the Random exploration module (used for the first n
        # timesteps).
        self.random_timesteps = random_timesteps
        self.random_exploration = SafeRandom(
            action_space, model=self.model, framework=self.framework, **kwargs
        )

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            np.array(0, np.int64),
            framework=self.framework,
            tf_name="timestep",
            dtype=np.int64,
        )
