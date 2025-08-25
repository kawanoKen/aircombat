# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os
import sys
import json
import time
import argparse
import importlib
import copy
import gymnasium as gym
import numpy as np
from ASRCAISim1.core import Factory, nljson
from ASRCAISim1.common import addPythonClass
from ASRCAISim1.GymManager import GymManager
from ASRCAISim1.plugins.rayUtility.utility.TransparentRemoteObject import unwrap_if_needed
from ASRCAISim1.plugins.MatchMaker.TwoTeamCombatMatchMaker import (
    getFromDesiredInitialState,
    getRandomizedInitialState,
    invert,
)
from R7ContestModels.R7ContestTwoTeamCombatMatchMaker import (
    getForEscort,
    R7ContestTwoTeamCombatMatchMaker
)

def managerConfigReplacer(config,matchInfo):
    """対戦カードを表すmatchInfoに従い、SimulationManagerのコンフィグを置き換える関数。
    configはSimulationManager::getManagerConfig()で得られるものであり、
    Simulationmanagerのコンストラクタに渡すenv_configのうち"Manager"キー以下の部分となる。
    """
    ret=copy.deepcopy(config)
    agentConfigDispatcher=ret["AgentConfigDispatcher"]
    assetConfigDispatcher=ret["AssetConfigDispatcher"]
    for team, info in matchInfo.get("teams",matchInfo).items():
        if team == "Factory":
            continue
        agentConfigDispatcher[team+"Agents"]=info["Agent"]
        if "InitialState" in info:
            num=len(assetConfigDispatcher[team+"InitialState"]["elements"])
            for i in range(num):
                assetConfigDispatcher[team+"InitialState"]["elements"][i]["value"]["instanceConfig"]=info["InitialState"][i]
    return ret
def factoryConfigReplacer(config,matchInfo):
    """対戦カードを表すmatchInfoに従い、SimulationManagerのFactory側のコンフィグを置き換える関数。
    configはSimulationManager::getFactoryModelConfig(True)で得られるものであり、
    Simulationmanagerのコンストラクタに渡すenv_configのうち"Factory"キー以下の部分となる。
    """
    ret=nljson(config)
    if "Factory" in matchInfo.get("common", {}):
        ret.merge_patch(matchInfo["common"]["Factory"])
    return ret

def wrapEnvForTwoTeamCombatMatchMaking(base_class):
    """MatchMakerの付帯情報をgym.Envに付加するラッパー
    """
    class WrappedEnvForTwoTeamCombatMatchMaking(base_class):
        def setMatch(self,matchInfo):
            self.matchInfo=matchInfo
            self.requestReconfigure(
                managerConfigReplacer(unwrap_if_needed(self.manager.getManagerConfig()()),matchInfo),
                factoryConfigReplacer(unwrap_if_needed(self.manager.getFactoryModelConfig(True)()),matchInfo)
            )
    return WrappedEnvForTwoTeamCombatMatchMaking

class R7ContestMatchMakerForEvaluation(R7ContestTwoTeamCombatMatchMaker):
    # R7年度コンテストの評価サーバ用のMatchMaker
    def initialize(self,config):
        super().initialize(config)
        self.teams = list(self.match_config["teams"].keys())
        self.team_config=self.match_config["teams"]
        self.initial_state_mode = self.match_config["initial_state"]
        if self.initial_state_mode =="selected":
            self.desired = self.match_config["desired"]
        self.number=4
        self.symmetric_initial_state=True
    def setAgent(self,ret,matchType,worker_index):
        for team in self.teams:
            userModelID=self.team_config[team]["userModelID"]
            isSingle=self.team_config[team]["isSingle"]
            if(isSingle):
                # 1個のインスタンスで1機なので異なるnameを割り当て、portはどちらも"0"
                ret[team]={"Agent":{
                    "type": "group",
                    "order": "fixed",
                    "elements": [
                        {"type": "External", "model": "Agent_"+userModelID, "policy": "Policy_" +
                            userModelID, "name": team+"_"+userModelID+"_"+str(i+1), "port": "0"} for i in range(self.number)
                    ]
                }}
            else:
                # 1個のインスタンスで2機分なので同じnameを割り当てることで1個のインスタンスとし、それぞれ異なるport("0"からstr(self.number-1))を割り当てる
                ret[team]={"Agent":{
                    "type": "group",
                    "order": "fixed",
                    "elements": [
                        {"type": "External", "model": "Agent_"+userModelID,
                            "policy": "Policy_"+userModelID, "name": team+"_"+userModelID, "port": str(i)} for i in range(self.number)
                    ]
                }}
        return ret
    def setInitialState(self,ret,matchType,worker_index):
        lower_bound={
            "pos": np.array([-20000.0,95000.0,-12000.0]),
            "vel": 260.0,
            "heading": 225.0
        }
        upper_bound={
            "pos": np.array([20000.0,105000.0,-6000.0]),
            "vel": 280.0,
            "heading": 315.0
        }
        if self.youth:
            lower_bound["pos"][2]=-10000.0
            upper_bound["pos"][2]=-10000.0
        if self.initial_state_mode == "random":
            initialStates=getRandomizedInitialState(
                self.teams,
                lower_bound,
                upper_bound,
                self.number,
                self.symmetric_initial_state
            )
            for team in self.teams:
                ret[team]["InitialState"]=initialStates[team]["InitialState"]
            ret[self.teams[0]]["InitialState"].append(getForEscort(np.random.rand()))
            ret[self.teams[1]]["InitialState"].append(invert(getForEscort(np.random.rand())))
        elif self.initial_state_mode =="selected":
            default = [
                {"pos":[-10000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
                {"pos":[10000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
                {"pos":[-20000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
                {"pos":[20000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
            ]
            initialStates=getFromDesiredInitialState(
                self.teams,
                self.desired,
                lower_bound,
                upper_bound,
                default,
                self.number
            )
            for team in self.teams:
                ret[team]["InitialState"]=initialStates[team]["InitialState"]
            ret[self.teams[0]]["InitialState"].append(getForEscort(np.random.rand()))
            ret[self.teams[1]]["InitialState"].append(invert(getForEscort(np.random.rand())))
        else: # fixed (元のjsonファイルに書かれたとおりの初期条件を用いる)
            pass
        return ret

def run(match_config):
    seed=match_config.get("seed",None)
    if(seed is None):
        import numpy as np
        seed = np.random.randint(2**31)
    logName="_vs_".join([info.get("logName",info["userModelID"]) for info in match_config["teams"].values()])
    # ユーザーモジュールの読み込み
    importedUserModules={}
    for team,info in match_config["teams"].items():
        userModelID=info["userModelID"]
        userModuleID=info["userModuleID"]
        args=info.get("args",{})
        args["userModelID"]=userModelID
        info["args"]=args
        if(not userModuleID in importedUserModules):
            try:
                module = importlib.import_module(userModuleID)
                assert hasattr(module, "getUserAgentClass")
                assert hasattr(module, "getUserAgentModelConfig")
                assert hasattr(module, "isUserAgentSingleAsset")
                assert hasattr(module, "getUserPolicy")
                if match_config["initial_state"]=="selected":
                    assert hasattr(module, "getBlueInitialState")
            except Exception as e:
                raise e  # 読み込み失敗時の扱いは要検討
            importedUserModules[userModuleID]=module
        module=importedUserModules[userModuleID]
        agentClass = module.getUserAgentClass(args)
        addPythonClass("Agent", "Agent_"+userModelID, agentClass)
        Factory.addDefaultModel(
            "Agent",
            "Agent_"+userModelID,
            {
                "class": "Agent_"+userModelID,
                "config": module.getUserAgentModelConfig(args)
            }
        )
    # コンフィグの生成
    loggerConfig = {}
    if "log_dir" in match_config:
        loggerConfig = {
            "MultiEpisodeLogger":{
                "class":"MultiEpisodeLogger",
                "config":{
                    "prefix":os.path.join(match_config["log_dir"],logName),
                    "episodeInterval":1,
                    "ratingDenominator":100
                }
            }
        }
        if match_config["replay"]:
            loggerConfig["GUIStateLogger"]={
                "class":"GUIStateLogger",
                "config":{
                    "prefix":os.path.join(match_config["log_dir"],logName),
                    "episodeInterval":1,
                    "innerInterval":1
                }
            }

    division="youth" if match_config["youth"] else "open"
    configs = [
        os.path.join(os.path.dirname(__file__), "common/R7_contest_"+division+"_mission_config.json"),
        os.path.join(os.path.dirname(__file__), "common/R7_contest_"+division+"_asset_placeholder.json"),
        {
            "Manager": {
                "Rewards": [],
                "seed":seed,
                "ViewerType":"R7Contest"+("Youth" if match_config["youth"] else "Open")+"God" if match_config["visualize"] else "None",
                "Loggers": loggerConfig,
                "AgentConfigDispatcher": {}
            }
        }
    ]
    context = {
        "config": configs,
        "worker_index": 0,
        "vector_index": 0
    }
    # 環境の生成
    env = wrapEnvForTwoTeamCombatMatchMaking(GymManager)(context)
    for team,info in match_config["teams"].items():
        info["isSingle"]=importedUserModules[info["userModuleID"]].isUserAgentSingleAsset(info["args"])
    if match_config["initial_state"]=="selected":
        match_config["desired"] = {
            team: importedUserModules[info["userModuleID"]].getBlueInitialState(info["args"])
            for team,info in match_config["teams"].items()
        }
    match_config["initial_state_number"]=5
    if not match_config["youth"]:
        match_config["symmetric_randomization"]=True
        match_config["heterogeneous_randomization"]=True
        match_config["shuffle_fighter_order"]=True
        match_config["randomized_asset_spec"]={
            "rcs_scale": [0.4,1.0], # 探知距離にして約0.8〜1.0倍
            "radar_range": [0.8,1.2],
            "radar_coverage": [60.0,120.0],
            "maximum_speed": [0.8,1.2],
            "num_missiles": [4,8],
            "missile_thrust": [0.8,1.2],
            "shot_approval_delay": [1,5],
        }
    matchMaker = R7ContestMatchMakerForEvaluation({"match_config":match_config})

    # StandalonePolicyの生成
    policies = {
        "Policy_"+info["userModelID"]: importedUserModules[info["userModuleID"]].getUserPolicy(info["args"])
        for team,info in match_config["teams"].items()
    }
    # policyMapperの定義(基本はデフォルト通り)

    def policyMapper(fullName):
        agentName, modelName, policyName = fullName.split(":")
        return policyName

    # 生成状況の確認
    matchInfo=matchMaker.makeNextMatch("",0)
    env.setMatch(matchInfo)
    observation_space = env.observation_space
    action_space = env.action_space
    print("=====Agent classes=====")
    for team,info in match_config["teams"].items():
        print("Agent_"+info["userModelID"], " = ", importedUserModules[info["userModuleID"]].getUserAgentClass(info["args"]))
    print("=====Policies=====")
    for name, policy in policies.items():
        print(name, " = ", type(policy))
    print("=====Policy Map (at reset)=====")
    for fullName in action_space:
        print(fullName, " -> ", policyMapper(fullName))
    print("=====Agent to Asset map=====")
    for agent in [a() for a in env.manager.getAgents()]:
        print(agent.getFullName(), " -> ", "{")
        for port, parent in agent.parents.items():
            print("  ", port, " : ", parent.getFullName())
        print("}")

    # シミュレーションの実行
    print("=====running simulation(s)=====")
    numEpisodes = match_config["num_episodes"]
    for episodeCount in range(numEpisodes):
        start=time.time()
        if episodeCount > 0 and episodeCount % 2 == 0:
            # 偶数回目は初期配置や機体性能も再生成
            matchInfo=matchMaker.makeNextMatch("",0)
            env.setMatch(matchInfo)
        elif episodeCount % 2 == 1:
            # 奇数回目はAgentだけ入れ替えてもう一回
            teams=list(matchInfo["teams"].keys())
            tmp=matchInfo["teams"][teams[0]]["Agent"]
            matchInfo["teams"][teams[0]]["Agent"]=matchInfo["teams"][teams[1]]["Agent"]
            matchInfo["teams"][teams[1]]["Agent"]=tmp
            env.setMatch(matchInfo)
        obs, info = env.reset()
        rewards = {k: 0.0 for k in obs.keys()}
        terminateds = {k: False for k in obs.keys()}
        truncateds = {k: False for k in obs.keys()}
        infos = {k: None for k in obs.keys()}
        for p in policies.values():
            p.reset()
        terminateds["__all__"]=False
        truncateds["__all__"]=False
        while not (terminateds["__all__"] or truncateds["__all__"]):
            observation_space = env.get_observation_space()
            action_space = env.get_action_space()
            actions = {k: policies[policyMapper(k)].step(
                o,
                rewards[k],
                terminateds[k] or truncateds[k],
                infos[k],
                k,
                observation_space[k],
                action_space[k]
            ) for k, o in obs.items() if policyMapper(k) in policies}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
        took=time.time()-start
        print("episode(", episodeCount+1, "/", numEpisodes, "), winner=",
              env.manager.getRuler()().winner, ", scores=", {k: v for k, v in env.manager.scores.items()}, ", calc. time=",took,"sec",", sim. time=",env.manager.getElapsedTime(),"sec")


if __name__ == "__main__":
    candidates=json.load(open("candidates.json","r"))
    parser=argparse.ArgumentParser()
    parser.add_argument("Blue",type=str,help="name of Blue team")
    parser.add_argument("Red",type=str,help="name of Red team")
    parser.add_argument("-n","--num_episodes",type=int,default=1,help="number of evaluation episodes")
    parser.add_argument("-l","--log_dir",type=str,help="log directory")
    parser.add_argument("-r", "--replay",action="store_true",help="use when you want to record episodes for later visualization")
    parser.add_argument("-v","--visualize",action="store_true",help="use when you want to visualize episodes")
    parser.add_argument("-i","--initial_state",type=str,default="random",help="initial state condition (random, selected, or fixed)")
    parser.add_argument("-y","--youth",action="store_true",help="use when you want to evaluate models for youth-division")
    args=parser.parse_args()
    assert(args.Blue in candidates and args.Red in candidates)
    match_config={
        "teams":{
            "Blue":{"userModelID":args.Blue},
            "Red":{"userModelID":args.Red},
        },
        "num_episodes":args.num_episodes,
        "replay":args.replay,
        "visualize":args.visualize,
        "initial_state":args.initial_state,
        "youth":args.youth,
    }
    for team,value in match_config["teams"].items():
        match_config["teams"][team].update(candidates[value["userModelID"]])
    if(args.log_dir is not None):
        match_config["log_dir"]=args.log_dir
    run(match_config)
