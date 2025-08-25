"""
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

A standalone version of RLlib Policy independent of Trainer.
The code for builing Policy instances is derived from RolloutWorker._update_policy_map
which is defined in ray.rllib.evaluation.rollout_worker.py  as of ray 2.5.1.
"""
from gymnasium.spaces import Discrete, MultiDiscrete
import logging
import numpy as np
from collections.abc import (
    Mapping,
    Sequence,
)
import tree  # pip install dm_tree
from typing import (
    TYPE_CHECKING,
)

import ray
ray_version = [int(s) for s in ray.__version__.split('.')]

from ray.rllib.connectors.agent.synced_filter import SyncedFilterAgentConnector
from ray.rllib.connectors.util import (
    create_connectors_for_policy,
)
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.filter import NoFilter, get_filter
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.policy import create_policy_for_framework, validate_policy_id
from ray.rllib.utils.tf_utils import (
    get_gpu_devices as get_tf_gpu_devices,
    get_tf_eager_cls_if_necessary,
)
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.utils.typing import (
    EnvObsType,
)
from ray.util.debug import log_once
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ASRCAISim1.policy.StandalonePolicy import StandalonePolicy
from ...utility.common import loadPolicyWeights

#tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

class StandaloneRayPolicy(StandalonePolicy):
    """Ray.RLLibのPolicyをTrainerから独立して動かすためのクラス。
    """
    def __init__(self,policyName,config,isLocal=False,isDeterministic=False):
        trainer_config=config["trainer_config"]
        self.name=policyName
        self.seed=config.get("seed",None)
        self.policy_class=config["policy_class"]
        self.weight=config.get("weight",None)
        self.policy_spec_config=config.get("policy_spec_config",{})
        self._remote_config=trainer_config
        self._local_config=trainer_config.update_from_dict(
            {"tf_session_args": trainer_config["local_tf_session_args"]}
        )
        self.preprocessing_enabled=True
        self.isLocal=isLocal
        self.isDeterministic=isDeterministic
        self.policy=None
        self.collector=None
    def _build(self,policy_spec,base_config):
        # This method was derived from
        # ray.rllib.evaluation.rollout_worker.RolloutWorker._update_policy_map
        # and related member methods.

        # This part was derived from
        # ray.rllib.evaluation.rollout_worker.RolloutWorker._get_complete_policy_specs_dict
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        logger.debug("Creating policy for {}".format(self.name))
        # Policy brings its own complete AlgorithmConfig -> Use it for this policy.
        if isinstance(policy_spec.config, AlgorithmConfig):
            merged_conf = policy_spec.config
        else:
            # Update the general config with the specific config
            # for this particular policy.
            merged_conf: "AlgorithmConfig" = base_config.copy(copy_frozen=False)
            merged_conf.update_from_dict(policy_spec.config or {})

        # Update num_workers and worker_index.
        merged_conf.worker_index = 0

        # Preprocessors.
        obs_space = policy_spec.observation_space
        # Initialize preprocessor for this policy to None.
        self.preprocessor = None
        if self.preprocessing_enabled:
            # Policies should deal with preprocessed (automatically flattened)
            # observations if preprocessing is enabled.
            if ray_version < [2,9,0]:
                include_multi_binary=base_config.get(
                    "_enable_rl_module_api", False
                )
            elif ray_version < [2,21,0]:
                include_multi_binary=base_config.get(
                    "_enable_new_api_stack", False
                )
            elif ray_version < [2,38,0]:
                include_multi_binary=base_config.get(
                    "enable_rl_module_and_learner", False
                )
            else: # as of 2.45.0
                include_multi_binary = False
            preprocessor = ModelCatalog.get_preprocessor_for_space(
                obs_space,
                merged_conf.model,
                include_multi_binary=include_multi_binary,
            )
            # Original observation space should be accessible at
            # obs_space.original_space after this step.
            if preprocessor is not None:
                obs_space = preprocessor.observation_space

            if not merged_conf.enable_connectors:
                # If connectors are not enabled, rollout worker will handle
                # the running of these preprocessors.
                self.preprocessor = preprocessor

        policy_spec.config = merged_conf
        policy_spec.observation_space = obs_space

        # This part was derived from
        # ray.rllib.evaluation.rollout_worker.RolloutWorker._build_policy_map
        self.policy = create_policy_for_framework(
            policy_id=self.name,
            policy_class=get_tf_eager_cls_if_necessary(
                policy_spec.policy_class, policy_spec.config
            ),
            merged_config=policy_spec.config,
            observation_space=policy_spec.observation_space,
            action_space=policy_spec.action_space,
            worker_index=0,
            seed=self.seed,
        )

        # Initialize the filter dict
        # This part was derived from
        # ray.rllib.evaluation.rollout_worker.RolloutWorker._update_filter_dict
        self.filter = NoFilter()
        if ray_version >= [2,39,0] or policy_spec.config.enable_connectors:
            if (
                self.policy.agent_connectors is None
                or self.policy.action_connectors is None
            ):
                create_connectors_for_policy(self.policy, policy_spec.config)
            #maybe_get_filters_for_syncing(self, self.name)
            if self.policy.agent_connectors:
                filter_connectors = self.policy.agent_connectors[SyncedFilterAgentConnector]
                # There can only be one filter at a time
                if filter_connectors:
                    assert len(filter_connectors) == 1, (
                        "ConnectorPipeline has multiple connectors of type "
                        "SyncedFilterAgentConnector but can only have one."
                    )
                    self.filter = filter_connectors[0].filter   
        else:
            filter_shape = tree.map_structure(
                lambda s: (
                    None
                    if isinstance(s, (Discrete, MultiDiscrete))  # noqa
                    else np.array(s.shape)
                ),
                self.policy.observation_space_struct,
            )

            self.filter = get_filter(
                policy_spec.config.observation_filter,
                filter_shape,
            )

        logger.info("Built policy: {}".format(self.policy))
        logger.info("Built preprocessor: {}".format(self.preprocessor))

        if(self.weight is not None):
            loadPolicyWeights(self.policy,self.weight)
    def reset(self):
        self.states={}
        self.prev_action={}
        self.prev_rewards={}
        self.agent_index={}
        self.collectors={}
        self.timeStep={}
    def step(self,observation,reward,done,info,agentFullName,observation_space,action_space):
        if(done):
            return None
        if(self.policy is None):
            policy_spec = PolicySpec(
                policy_class=self.policy_class,
                observation_space=observation_space,
                action_space=action_space,
                config=self.policy_spec_config
            )
            if(self.isLocal):
                self._build(policy_spec,self._local_config)
            else:
                self._build(policy_spec,self._remote_config)
        if(not agentFullName in self.agent_index):
            self.agent_index[agentFullName]=len(self.agent_index)
            self.timeStep[agentFullName]=0
        #preprocess and filter raw obs same as SamplerInput._process_observations
        prep_obs: EnvObsType = observation
        if self.preprocessor is not None:
            prep_obs = self.preprocessor.transform(observation)
            if log_once("prep_obs"):
                logger.info("Preprocessed obs: {}".format(summarize(prep_obs)))
        filtered_obs: EnvObsType = self.filter(
            prep_obs
        )
        if log_once("filtered_obs"):
            logger.info("Filtered obs: {}".format(summarize(filtered_obs)))

        if(self.timeStep[agentFullName]==0):
            #try:
            #    max_seq_len = policy.config["model"]["max_seq_len"]
            #except KeyError:
            #    max_seq_len = 1
            self.collectors[agentFullName]=AgentCollector(
                self.policy.view_requirements,
                max_seq_len=1,
                disable_action_flattening=self.policy.config.get(
                    "_disable_action_flattening", False
                ),
                intial_states=self.policy.get_initial_state(),
                is_policy_recurrent=self.policy.is_recurrent(),
                is_training=False,
            )
            self.collectors[agentFullName].add_init_obs(
                episode_id=0,
                agent_index=self.agent_index[agentFullName],
                env_id=0,
                init_obs=filtered_obs,
                init_infos={},
                t=self.timeStep[agentFullName],
            )
        else:
            values={
                    SampleBatch.T: self.timeStep[agentFullName],
                    SampleBatch.ENV_ID: 0,
                    SampleBatch.AGENT_INDEX : self.agent_index[agentFullName],
                    SampleBatch.ACTIONS: self.prev_action[agentFullName],
                    SampleBatch.REWARDS: reward,
                    SampleBatch.DONES: done,
                    SampleBatch.NEXT_OBS: filtered_obs
                }
            for i,state in enumerate(self.states[agentFullName]):
                values["state_out_{}".format(i)]=state
            self.collectors[agentFullName].add_action_reward_next_obs(values)
        action,state_out,info=self.policy.compute_actions_from_input_dict(
            input_dict=self.collectors[agentFullName].build_for_inference(),
            explore=not self.isDeterministic,
            timestep=self.timeStep[agentFullName],
        )
        def remove_batch_dim(t):
            # compute_actions_from_input_dictの出力はバッチサイズ1の次元が先頭に付いているため取り除く必要がある。
            if isinstance(t, Sequence):
                return type(t)(remove_batch_dim(v) for v in t)
            elif isinstance(t, Mapping):
                return type(t)((k,remove_batch_dim(v)) for k, v in t.items())
            else:
                return t[0]
        self.states[agentFullName]=remove_batch_dim(state_out)
        action=remove_batch_dim(action)
        self.prev_action[agentFullName]=flatten_to_single_ndarray(action)
        self.timeStep[agentFullName]+=1
        return action
