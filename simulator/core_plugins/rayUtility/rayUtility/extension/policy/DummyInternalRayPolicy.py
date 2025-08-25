"""
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

A dummy RLlib Policy with minimum interface required for ray.rllib.policy.Policy
"""
import gymnasium as gym
import numpy as np
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    ModelWeights,
    TensorStructType,
    TensorType,
)

class DummyInternalRayPolicy(Policy):
    """
    InternalなAgentはobs,actionともにDiscrete(1)を設定しているので適当にsampleして返すだけ。
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: AlgorithmConfigDict,
    ):
        super().__init__(observation_space, action_space, config)

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return np.stack([self.action_space.sample() for _ in obs_batch],axis=0),[],{}

    @override(Policy)
    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        pass

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        pass

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        pass
