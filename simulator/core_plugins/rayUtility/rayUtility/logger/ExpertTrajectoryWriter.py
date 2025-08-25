# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
import sys,os,time
import numpy as np
from datetime import datetime
from ASRCAISim1.core import (
	Callback,
	getValueFromJsonKRD,
	SimpleMultiPortCombiner,
)
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_writer import JsonWriter

class ExpertTrajectoryWriter(Callback):
	"""異なるobservation,action形式を持つAgentを摸倣する際のTrajectory記録用のLogger。
	RLlibのSampleBatchBuilderを使用する。
	Managerのconfigで指定することを要求する。
	"""
	def initialize(self):
		super().initialize()
		self.prefix=getValueFromJsonKRD(self.modelConfig,"prefix",self.randomGen,"")
		self._enable_rl_module_api=self.modelConfig().get("_enable_rl_module_api", False)
		self._disable_preprocessor_api=self.modelConfig().get("_disable_preprocessor_api", False)
		self.preprocessing_enabled=not self._disable_preprocessor_api
		self.preprocessor_options=self.modelConfig().get("preprocessor_options", None) # Only used for GenericPixelPreprocessor
		self.disable_action_flattening=self.modelConfig().get("disable_action_flattening", True)
		self.episodeCounter=-1
		self.writers={}
		self.view_reqs={}
		self.agentCollectors={}
		self.preprocessors={}
		self.dones={}
		self.agentIndices={}
		self.agentIndexCount=0
	def makeWriter(self,identifier,observation_space,action_space):
		if(len(self.prefix)==0 or self.prefix[-1]=='/'):
			path=self.prefix+identifier
		else:
			path=self.prefix+"_"+identifier
		os.makedirs(os.path.dirname(path),exist_ok=True)
		ioctx=IOContext(
			log_dir=path,
			config={},
			worker_index=self.manager.worker_index
		)
		self.writers[identifier]=JsonWriter(path,ioctx)
		self.preprocessors[identifier]=ModelCatalog.get_preprocessor_for_space(
			observation_space,
			self.preprocessor_options,
			include_multi_binary=self._enable_rl_module_api,
		)
		self.view_reqs[identifier]={
			SampleBatch.OBS: ViewRequirement(space=self.preprocessors[identifier].observation_space),
			SampleBatch.NEXT_OBS: ViewRequirement(
				data_col=SampleBatch.OBS,
				shift=1,
				space=self.preprocessors[identifier].observation_space),
			SampleBatch.ACTIONS:ViewRequirement(
				space=action_space, used_for_compute_actions=False),
			#SampleBatch.PREV_ACTIONS: ViewRequirement(
			#	data_col=SampleBatch.ACTIONS,
			#	shift=-1,
			#	space=action_space),
			SampleBatch.REWARDS: ViewRequirement(),
			SampleBatch.PREV_REWARDS: ViewRequirement(
				data_col=SampleBatch.REWARDS, shift=-1),
			SampleBatch.DONES: ViewRequirement(),
			SampleBatch.ENV_ID: ViewRequirement(),
			SampleBatch.EPS_ID: ViewRequirement(),
			SampleBatch.UNROLL_ID: ViewRequirement(),
			SampleBatch.AGENT_INDEX: ViewRequirement(),
			SampleBatch.ACTION_PROB: ViewRequirement(),
			SampleBatch.ACTION_LOGP: ViewRequirement(),
			SampleBatch.T: ViewRequirement(),
		}
	def parseInitialObs(self,identifier,observation_space,action_space,obs):
		prep_obs=obs
		if self.preprocessing_enabled:
			prep_obs=self.preprocessors[identifier].transform(obs)
		self.agentCollectors[identifier]=AgentCollector(
			self.view_reqs[identifier],
			max_seq_len=1,
			disable_action_flattening=self.disable_action_flattening,
			intial_states=None,
			is_policy_recurrent=False,
			is_training=True,
		)
		self.agentCollectors[identifier].add_init_obs(
			episode_id=self.episodeCounter,
			agent_index=self.agentIndices[identifier],
			env_id=self.manager.vector_index,
			init_obs=prep_obs,
			init_infos={},
			t=-1,
		)
		self.dones[identifier]=False
	def addStepData(self,identifier,obs,action,reward):
		prep_obs=obs
		if self.preprocessing_enabled:
			prep_obs=self.preprocessors[identifier].transform(obs)
		self.agentCollectors[identifier].add_action_reward_next_obs(
			{
				SampleBatch.T:self.manager.getExposedStepCount(),
				SampleBatch.ENV_ID:self.manager.vector_index,
				SampleBatch.AGENT_INDEX:self.agentIndices[identifier],
				SampleBatch.ACTIONS:action,
				SampleBatch.REWARDS:reward,
				SampleBatch.DONES:self.dones[identifier],
				SampleBatch.NEXT_OBS:prep_obs,
				SampleBatch.ACTION_PROB:1.0,
				SampleBatch.ACTION_LOGP:0.0
			}
		)
	def onEpisodeBegin(self):
		self.episodeCounter+=1
		self.agentIndices={}
		self.agentIndexCount=0
		for agent in self.manager.experts.values():
			agent=agent()
			identifier_base=agent.trajectoryIdentifier
			if(isinstance(agent.imitator,SimpleMultiPortCombiner)):
				childrenKeys=list(agent.imitator.children.keys())
				observation_space=self.manager.observation_space[agent.getFullName()][0]
				action_space=self.manager.action_space[agent.getFullName()][0]
				obs=agent.imitatorObs
				for idx in range(len(childrenKeys)):
					identifier=identifier_base+str(idx+1)
					key=childrenKeys[idx]
					if(not identifier in self.agentIndices):
						self.agentIndices[identifier]=self.agentIndexCount
						self.agentIndexCount+=1
					if(not identifier in self.writers):
						self.makeWriter(
							identifier,
							observation_space[key],
							action_space[key]
						)
					self.parseInitialObs(identifier,observation_space[key],action_space[key],obs[key])
				self.dones[identifier_base]=False
			else:
				identifier=identifier_base
				observation_space=self.manager.observation_space[agent.getFullName()][0]
				action_space=self.manager.action_space[agent.getFullName()][0]
				obs=agent.imitatorObs
				if(not identifier in self.agentIndices):
					self.agentIndices[identifier]=self.agentIndexCount
					self.agentIndexCount+=1
				if(not identifier in self.writers):
					self.makeWriter(
						identifier,
						observation_space,
						action_space
					)
				self.parseInitialObs(identifier,observation_space,action_space,obs)
	def onStepEnd(self):
		for agent in self.manager.experts.values():
			agent=agent()
			if agent in [a() for a in self.manager.agentsToAct.values()]:
				identifier_base=agent.trajectoryIdentifier
				if(isinstance(agent.imitator,SimpleMultiPortCombiner)):
					childrenKeys=list(agent.imitator.children.keys())
					done=self.manager.dones[agent.getFullName()]
					if(not self.dones[identifier_base]):
						self.dones[identifier_base]=done
						obs=agent.imitatorObs
						action=agent.imitatorAction
						rewards=self.manager.rewards[agent.getFullName()]
						reward=sum(rewards.values())
						for idx in range(len(childrenKeys)):
							identifier=identifier_base+str(idx+1)
							key=childrenKeys[idx]
							if(not self.dones[identifier]):
								self.dones[identifier]=done or not obs["isAlive"][key]
								self.addStepData(identifier,obs[key],action[key],reward/len(childrenKeys))
				else:
					identifier=identifier_base
					done=self.manager.dones[agent.getFullName()]
					if(not self.dones[identifier]):
						self.dones[identifier]=done
						obs=agent.imitatorObs
						action=agent.imitatorAction
						rewards=self.manager.rewards[agent.getFullName()]
						reward=sum(rewards.values())
						self.addStepData(identifier,obs,action,reward)
	def onEpisodeEnd(self):
		for identifier in self.writers:
			self.writers[identifier].write(self.agentCollectors[identifier].build_for_training(self.view_reqs[identifier]))
