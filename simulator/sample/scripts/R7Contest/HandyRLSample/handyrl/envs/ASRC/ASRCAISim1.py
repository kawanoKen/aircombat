# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os
import random
import itertools
import copy
import collections

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...util import map_r
from ...environment import BaseEnvironment
from ASRCAISim1.plugins.HandyRLUtility.distribution import getDefaultLegalActions

from ASRCAISim1.core import ExpertWrapper, SimpleMultiPortCombiner
from ASRCAISim1.GymManager import GymManager, getDefaultPolicyMapper
from ASRCAISim1.plugins.MatchMaker.TwoTeamCombatMatchMaker import wrapEnvForTwoTeamCombatMatchMaking, managerConfigReplacer
from ASRCAISim1.plugins.Supervised.libSupervised import UnsupervisedTaskMixer, SupervisedTaskMixer

def detach(src):
    return map_r(src, copy.deepcopy)

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        original_overrider=args["env_config"].get("overrider",lambda c,w,v:c)
        def overrider(config,worker_index,vector_index):
            nw=1024
            if("seed" in config["Manager"]):
                config["Manager"]["seed"]=config["Manager"]["seed"]+worker_index+nw*vector_index
            config=original_overrider(config,worker_index,vector_index)
            return config
        self.env_config=args["env_config"]
        context = copy.deepcopy(self.env_config)
        worker_index=args.get("id",-1)+1
        context.update({
            "overrider": overrider,
            "worker_index": worker_index,
            "vector_index": 0 if worker_index>0 else -2
        })
        cwd=args["env_config"].get("cwd",".")
        if isinstance(context["config"],str):
            context["config"]=os.path.join(cwd,context["config"])
        elif isinstance(context["config"],collections.abc.Sequence):
            context["config"]=[
                os.path.join(cwd,c) if isinstance(c,str) else c
                for c in context["config"]
            ]
        self.env = wrapEnvForTwoTeamCombatMatchMaking(GymManager)(context)
        self.completeEnvConfig={
            key:value for key,value in self.env_config.items() if key!="config"
        }
        self.completeEnvConfig["config"]={
                "Manager":self.env.getManagerConfig()(),
                "Factory":self.env.getFactoryModelConfig()()
            }
        self.teams=args['teams']
        self.policy_config = args["policy_config"]
        self._policySetup()
    def reset(self, args={}):
        obs, info = self.env.reset()
        self.agentConfig={}
        self.next_actors_full={}
        self.last_obs={}
        self.current_obs={}
        idx=0
        for agent in self.env.manager.getAgents():
            agent=agent()
            key=agent.getFullName()
            agentName,modelName,policyName=key.split(":")
            if agentName.endswith("E_Agent"):
                continue
            policyName=self.policyMapper(key)
            policy_config=self.policy_config[self.policy_base[policyName]]
            self.agentConfig[key]={
                "team":agent.getTeam(),
                "isExpertWrapper":isinstance(agent,ExpertWrapper),
                "isSimpleMultiPortCombiner":{"Expert":False,"Imitator":False},
                "splitExpert":False,
                "expertChildrenKeys":[],
                "splitImitator":False,
                "imitatorChildrenKeys":[],
                "policyName":policyName,
                "playerIndex":[idx],
                "imitatorIndex":[]
            }
            if(self.agentConfig[key]["isExpertWrapper"]):
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["splitExpert"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"] and not policy_config.get("multi_port",False)
                self.agentConfig[key]["splitImitator"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"] and policy_config.get("split_imitator",False)
                if(self.agentConfig[key]["splitExpert"]):
                    self.agentConfig[key]["playerIndex"]=list(range(idx,idx+len(agent.expert.children)))
                    self.agentConfig[key]["expertChildrenKeys"]=list(agent.expert.children.keys())
            idx+=len(self.agentConfig[key]["playerIndex"])
        for agent in self.env.manager.getAgents():
            agent=agent()
            key=agent.getFullName()
            agentName,modelName,policyName=key.split(":")
            if agentName.endswith("E_Agent"):
                continue
            if(self.agentConfig[key]["isExpertWrapper"]):
                if(self.agentConfig[key]["splitImitator"]):
                    self.agentConfig[key]["imitatorIndex"]=list(range(idx,idx+len(agent.imitator.children)))
                    self.agentConfig[key]["imitatorChildrenKeys"]=list(agent.imitator.children.keys())
                else:
                    self.agentConfig[key]["imitatorIndex"]=[idx]
            idx+=len(self.agentConfig[key]["imitatorIndex"])
        self.player_list = sum([
            [(k,i) for i in range(len(c["playerIndex"]))] for k,c in self.agentConfig.items()
        ],[])
        self.num_actual_players =len(self.player_list)
        self.policy_map = [self.agentConfig[key]["policyName"] for key,idx in self.player_list]
        self.teams = [self.agentConfig[key]["team"] for key,idx in self.player_list]
        #imitatorの付加
        self.imitator_list = sum([
            [(k,i) for i in range(len(c["imitatorIndex"]))] for k,c in self.agentConfig.items()
        ],[])
        self.player_list.extend(self.imitator_list)
        self.policy_map.extend(["Imitator" for i in self.imitator_list])
        self.teams.extend([self.agentConfig[key]["team"] for key,idx in self.imitator_list])
        self._parse_obs_dones(obs,self.env.manager.dones,self.env.manager.dones)
    def _parse_obs_dones(self,obs,terminateds,truncateds):
        self.dones = {}
        for key, d in terminateds.items():
            if(key=="__all__"):
                self.all_done=d
            else:
                agentName,modelName,policyName=key.split(":")
                if agentName.endswith("E_Agent"):
                    continue
                c=self.agentConfig[key]
                for idx in c["playerIndex"]+c["imitatorIndex"]:
                    self.dones[idx]=d
        for key, d in truncateds.items():
            if(key=="__all__"):
                self.all_done = self.all_done or d
            else:
                agentName,modelName,policyName=key.split(":")
                if agentName.endswith("E_Agent"):
                    continue
                c=self.agentConfig[key]
                for idx in c["playerIndex"]+c["imitatorIndex"]:
                    self.dones[idx] = self.dones[idx] or d
        self.last_obs=self.current_obs
        self.current_obs={}
        for key, o in obs.items():
            agentName,modelName,policyName=key.split(":")
            if agentName.endswith("E_Agent"):
                continue
            c=self.agentConfig[key]
            if(c["isExpertWrapper"]):
                imObs=o[0]
                if(self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"]):
                    imIsAlive=imObs.pop("isAlive")
                if(c["splitImitator"]):
                    for idx,sub in zip(c["imitatorIndex"],c["imitatorChildrenKeys"]):
                        self.current_obs[idx]=imObs[sub]
                        self.dones[idx]=self.dones[idx] or not imIsAlive[sub]
                else:
                    self.current_obs[c["imitatorIndex"][0]]=imObs
                exObs=o[1]
                if(self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"]):
                    exIsAlive=exObs.pop("isAlive")
                if(c["splitExpert"]):
                    for idx,sub in zip(c["playerIndex"],c["expertChildrenKeys"]):
                        self.current_obs[idx]=exObs[sub]
                        self.dones[idx]=self.dones[idx] or not exIsAlive[sub]
                else:
                    self.current_obs[c["playerIndex"][0]]=exObs
            else:
                self.current_obs[c["playerIndex"][0]]=o
        assert(set(self.dones.keys())==set(self.current_obs.keys()))
        self.last_actors_full = self.next_actors_full
        self.next_actors_full = {idx: not self.dones[idx] for idx in self.current_obs}
    def _parse_rewards(self,rewards):
        self.rewards={}
        for key, r in rewards.items():
            agentName,modelName,policyName=key.split(":")
            if agentName.endswith("E_Agent"):
                continue
            c=self.agentConfig[key]
            for idx in c["playerIndex"]:
                self.rewards[idx]=r/len(c["playerIndex"])
            for idx in c["imitatorIndex"]:
                self.rewards[idx]=r/len(c["imitatorIndex"])
    def _gather_actions(self,actions):
        ret={}
        for idx, action in actions.items():
            if self.policy_map[idx]!="Imitator":
                key,_ = self.player_list[idx]
                c=self.agentConfig[key]
                ret[key]=action
                if(c["splitExpert"]):
                    for ck, sub in zip(c["expertChildrenKeys"],action):
                        ret[key][ck] = sub
        return ret
    def step(self, actions):
        self.imitator_legal_actions={}
        for key,idx in self.imitator_list:
            c=self.agentConfig[key]
            self.imitator_legal_actions[c["imitatorIndex"][idx]]=self.legal_actions(c["imitatorIndex"][idx])
        try:
            obs, rewards, terminateds, truncateds, infos = self.env.step(self._gather_actions(actions))
        except Exception as e:
            print(self.env.manager.getElapsedTime(),",",actions)
            raise e
        self._parse_rewards(rewards)
        self._parse_obs_dones(obs,terminateds,truncateds)

    def turns(self):
        return [idx for idx, is_next_actor in self.next_actors_full.items() if is_next_actor and self.policy_map[idx]!="Imitator"]

    def terminal(self):
        return self.all_done

    def reward(self):
        return {idx: self.rewards.get(idx,0.0) for idx, key in enumerate(self.player_list)}

    def outcome(self):
        winner = self.env.manager.getRuler()().winner
        return {idx: 0 if winner =="" else (+1 if self.teams[idx] == winner else -1)
            for idx, key in enumerate(self.player_list)
        }
    def score(self):
        scores = self.env.manager.scores
        return {idx: scores[self.teams[idx]]
            for idx, key in enumerate(self.player_list)
        }

    def legal_actions(self, player):
        ac_space=self.action_space(player)
        return getDefaultLegalActions(ac_space)

    def players(self):
        #maximum number of players
        return list(range(len(self.player_list)))

    def observation(self, player=None):
        return self.current_obs[player]

    def net(self, policyName):
        return self.policy_config[policyName]["model_class"]

    #ASRC
    def _observation_space_internal(self, manager, agentFullName, asImitator=False, splitIndex=0):
        space=manager.get_observation_space()[agentFullName]
        c=self.agentConfig[agentFullName]
        if(c["isExpertWrapper"]):
            if(asImitator):
                if(c["splitImitator"]):
                    return space[0][c["imitatorChildrenKeys"][splitIndex]]
                else:
                    return space[0]
            else:
                if(c["splitExpert"]):
                    return space[1][c["expertChildrenKeys"][splitIndex]]
                else:
                    return space[1]
        else:
            return space
    def _action_space_internal(self, manager, agentFullName, asImitator=False, splitIndex=0):
        space=manager.get_action_space()[agentFullName]
        c=self.agentConfig[agentFullName]
        if(c["isExpertWrapper"]):
            if(asImitator):
                if(c["splitImitator"]):
                    return space[0][c["imitatorChildrenKeys"][splitIndex]]
                else:
                    return space[0]
            else:
                if(c["splitExpert"]):
                    return space[1][c["expertChildrenKeys"][splitIndex]]
                else:
                    return space[1]
        else:
            return space
    def observation_space(self, player):
        asImitator = player >= self.num_actual_players
        agentFullName, splitIndex = self.player_list[player]
        return self._observation_space_internal(self.env.manager, agentFullName, asImitator, splitIndex)
    def action_space(self, player):
        asImitator = player >= self.num_actual_players
        agentFullName, splitIndex = self.player_list[player]
        return self._action_space_internal(self.env.manager, agentFullName, asImitator, splitIndex)
    def setMatch(self,matchInfo):
        self.policy_base={
            info["Policy"]+info["Suffix"]: info["Policy"]
            for info in matchInfo.get("teams",matchInfo).values()
        }
        self.env.setMatch(matchInfo)
    def _policySetup(self):
        dummyEnv=None
        for policyName,policy_config in self.policy_config.items():
            dummyContext = copy.deepcopy(self.completeEnvConfig)
            dummyContext["config"]["Manager"]["Viewer"] = "None"
            dummyContext["config"]["Manager"]["Loggers"] = {}
            isMultiPort = policy_config.get("multi_port",False) is True
            matchInfo={
                "teams":{
                    team:{"Policy":policyName,"Suffix":"","Weight":-1,"MultiPort":isMultiPort}
                    for team in self.teams
                }
            }
            dummyContext["config"]["Manager"]=managerConfigReplacer(dummyContext["config"]["Manager"],matchInfo)
            dummyContext.update({
                "worker_index": -1,
                "vector_index": -1,
            })
            if dummyEnv is not None:
                dummyEnv.manager.purge()
            dummyEnv = GymManager(dummyContext)
            dummyEnv.reset()

            ac=dummyEnv.get_action_space()
            key=next(iter(ac))
            _,m_name,p_name=key.split(":")
            if(p_name==policyName):
                pass
            elif(p_name=="Internal"):
                pass
            else:
                raise ValueError("Invalid policy config.")
            agent=dummyEnv.manager.getAgent(key)()
            self.agentConfig={
                key:{
                    "isExpertWrapper":isinstance(agent,ExpertWrapper),
                    "isSimpleMultiPortCombiner":{"Expert":False,"Imitator":False},
                }
            }
            if(self.agentConfig[key]["isExpertWrapper"]):
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["splitExpert"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"] and not isMultiPort
                if(self.agentConfig[key]["splitExpert"]):
                    self.agentConfig[key]["expertChildrenKeys"]=list(agent.expert.children.keys())
                self.agentConfig[key]["splitImitator"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"] and policy_config.get("split_imitator",False)
                if(self.agentConfig[key]["splitImitator"]):
                    self.agentConfig[key]["imitatorChildrenKeys"]=list(agent.imitator.children.keys())
            self.policy_config[policyName]["observation_space"]=self._observation_space_internal(dummyEnv.manager, key, False, 0)
            self.policy_config[policyName]["action_space"]=self._action_space_internal(dummyEnv.manager, key, False, 0)
            aux_truth_space={}
            for v in dummyEnv.manager.getCallbacks():
                if(isinstance(v(),SupervisedTaskMixer)):
                    name=v().getName()
                    if(v().isTarget(agent)):
                        aux_truth_space[name]=detach(v().calcTaskTruthSpace(agent))
            self.policy_config[policyName]["auxiliary_task_truth_space"]=aux_truth_space

    def policyMapper(self,agentId):
        """
        エージェントのfullNameから対応するポリシー名を抽出する関数。
        agentId=agentName:modelName:policyName
        """
        agentName,modelName,policyName=agentId.split(":")
        info = None
        for team in self.teams:
            if(agentName.startswith(team)):
                info=self.env.matchInfo.get("teams",self.env.matchInfo)[team]
                break
        if(info is None):
            raise ValueError("Invalid agent name. fullName=",agentId)
        return info["Policy"]+info["Suffix"]
    def getImitationInfo(self):
        """ExpertWrapperを使用しているAgentについて、模倣する側の情報を返す。
        """
        ret={}
        for key,idx in self.imitator_list:
            agent=self.env.manager.getAgent(key)()
            c=self.agentConfig[key]
            if(self.last_actors_full.get(c["imitatorIndex"][idx],False)):
                imObs=detach(self.last_obs[c["imitatorIndex"][idx]])
                if(c["splitImitator"]):
                    imAction=detach(agent.imitatorAction[c["imitatorChildrenKeys"][idx]])
                else:
                    imAction=detach(agent.imitatorAction)
                imLegalActions=self.imitator_legal_actions[c["imitatorIndex"][idx]]
                ret[c["imitatorIndex"][idx]]={
                    "observation": imObs,
                    "action": imAction,
                    "legal_actions": imLegalActions,
                }
        return ret
    def getAuxiliaryTaskTruth(self):
        def convert(src):#キーをAgent名からplayer idに変換
            ret={}
            for key,s in src.items():
                c=self.agentConfig[key]
                agent=self.env.manager.getAgent(key)()
                if(c["isExpertWrapper"]):
                    for idx in c["imitatorIndex"]:
                        ret[idx]=s
                for idx in c["playerIndex"]:
                    ret[idx]=s
            return ret
        truths={}
        for v in self.env.manager.getCallbacks():
            if(isinstance(v(),SupervisedTaskMixer)):
                k=v().getName()
                truths[k]=[convert(t) for t in detach(v().truths)]
        return truths
