import random
import os
import time
import importlib
import copy
import traceback
import numpy as np
from timeout_decorator import timeout, TimeoutError

from ASRCAISim1.core import Factory, nljson, Fighter # type: ignore
from ASRCAISim1.common import addPythonClass
from ASRCAISim1.GymManager import GymManager
from ASRCAISim1.plugins.rayUtility.utility.TransparentRemoteObject import unwrap_if_needed # type: ignore
from ASRCAISim1.plugins.MatchMaker.TwoTeamCombatMatchMaker import ( # type: ignore
    getFromDesiredInitialState,
    getRandomizedInitialState,
    invert,
)
from R7ContestModels.R7ContestTwoTeamCombatMatchMaker import (
    getForEscort,
    R7ContestTwoTeamCombatMatchMaker
)

class Fight():
    def __init__(self, match_config: dict[str, dict]) -> None:
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
                    raise e
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
        self.importedUserModules = importedUserModules

        # コンフィグの生成
        loggerConfig = {}
        if "log_dir" in match_config:
            # loggerConfig = {
            #     "MultiEpisodeLogger":{
            #         "class":"MultiEpisodeLogger",
            #         "config":{
            #             "prefix":os.path.join(match_config["log_dir"],logName), # type: ignore
            #             "episodeInterval":1,
            #             "ratingDenominator":100
            #         }
            #     }
            # }
            if match_config["replay"]:
                loggerConfig["GUIStateLogger"]={
                    "class":"GUIStateLogger",
                    "config":{
                        "prefix":os.path.join(match_config["log_dir"],logName), # type: ignore
                        "episodeInterval":1,
                        "innerInterval":1
                    }
                }

        division="youth" if match_config["youth"] else "open"
        configs = [
            os.path.join(match_config["common_dir"], "R7_contest_"+division+"_mission_config.json"), # type: ignore
            os.path.join(match_config["common_dir"], "R7_contest_"+division+"_asset_placeholder.json"), # type: ignore
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

        self.context = context

        # 環境の生成
        self.env = wrapEnvForTwoTeamCombatMatchMaking(GymManager)(context)

        for team,info in match_config["teams"].items():
            info["isSingle"]=importedUserModules[info["userModuleID"]].isUserAgentSingleAsset(info["args"])
        if match_config["initial_state"]=="selected":
            match_config["desired"] = {
                team: importedUserModules[info["userModuleID"]].getBlueInitialState(info["args"])
                for team,info in match_config["teams"].items()
            }
        match_config["initial_state_number"]=5 # type: ignore
        if not match_config["youth"]:
            match_config["symmetric_randomization"]=True # type: ignore
            match_config["heterogeneous_randomization"]=True # type: ignore
            match_config["shuffle_fighter_order"]=True # type: ignore
            match_config["randomized_asset_spec"]={
                "rcs_scale": [0.4,1.0],
                "radar_range": [0.8,1.2],
                "radar_coverage": [60.0,120.0],
                "maximum_speed": [0.8,1.2],
                "num_missiles": [4,8],
                "missile_thrust": [0.8,1.2],
                "shot_approval_delay": [1,5],
            }

        self.match_config = match_config
        self.matchMaker = R7ContestMatchMakerForEvaluation({"match_config":match_config})


    def make_matchInfo(self) -> None:
        self.matchInfo=self.matchMaker.makeNextMatch("",0)


    def swap_condition(self) -> None:
        teams=list(self.matchInfo["teams"].keys())
        tmp=self.matchInfo["teams"][teams[0]]["Agent"]
        self.matchInfo["teams"][teams[0]]["Agent"]=self.matchInfo["teams"][teams[1]]["Agent"]
        self.matchInfo["teams"][teams[1]]["Agent"]=tmp


    def set_match(self) -> None:
        self.env.setMatch(self.matchInfo)


    def get_ticks(self, num_assets):
        ticks = {}
        red_fighter_names = [a().getFullName() for a in self.env.manager.getAgents() if 'Red' in a().getFullName()]
        blue_fighter_names = [a().getFullName() for a in self.env.manager.getAgents() if 'Blue' in a().getFullName()]
        ticks['Red'] = self.get_stepcount(red_fighter_names, num_assets)
        ticks['Blue'] = self.get_stepcount(blue_fighter_names, num_assets)

        return ticks


    def get_stepcount(self, fighter_names, num_assets):
        tick = {}
        if len(fighter_names)>1:
            for fighter_name in fighter_names:
                agentName, modelName, policyName = fighter_name.split(":")
                team = agentName.split('_')[0]
                port = agentName.split('_')[-1]
                tick[int(port)] = self.env.manager.getAgentStepCount(fighter_name)
        elif len(fighter_names) == 1:
            for i in range(num_assets):
                tick[i+1] = self.env.manager.getAgentStepCount(fighter_names[0])
        return tick


    def get_data(self, data, fighter_names):
        ptime = self.env.manager.getElapsedTime() #self.env.manager.getTime().__str__()
        data_f = {}
        i = 1
        for k, v in data.items():
            if k in fighter_names:
                data_f[i] = v
                i+=1
        out = {'ptime':ptime, 'data':data_f}
        n_data = len(data_f)
        return out, n_data


    def update_logs(self, red_fighter_names, blue_fighter_names, logs, obs, infos, red_actions, blue_actions):
        def conv(src):
            return map_r(src, lambda d: d.tolist() if isinstance(d, np.ndarray) else d)

        a, n_data = self.get_data(conv(red_actions), red_fighter_names)
        if n_data:
            logs['Red']['actions'].append(a)
        o, n_data = self.get_data(conv(obs), red_fighter_names)
        if n_data:
            logs['Red']['observations'].append(o)
        info, n_data = self.get_data(infos, red_fighter_names)
        if n_data:
            logs['Red']['infos'].append(info)

        a, n_data = self.get_data(conv(blue_actions), blue_fighter_names)
        if n_data:
            logs['Blue']['actions'].append(a)
        o, n_data = self.get_data(conv(obs), blue_fighter_names)
        if n_data:
            logs['Blue']['observations'].append(o)
        info, n_data = self.get_data(infos, blue_fighter_names)
        if n_data:
            logs['Blue']['infos'].append(info)


    def run(self, random_test: bool = False) -> tuple:
        if random_test:
            return random.choice(['Red', 'Blue', '']), {}

        # StandalonePolicyの生成
        policies = {
            "Policy_"+info["userModelID"]: self.importedUserModules[info["userModuleID"]].getUserPolicy(info["args"])
            for team,info in self.match_config["teams"].items()
        }

        # 生成状況の確認
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        print("=====Agent classes=====")
        for team,info in self.match_config["teams"].items():
            print("Agent_"+info["userModelID"], " = ", self.importedUserModules[info["userModuleID"]].getUserAgentClass(info["args"]))
        print("=====Policies=====")
        for name, policy in policies.items():
            print(name, " = ", type(policy))
        print("=====Policy Map (at reset)=====")
        for fullName in action_space:
            print(fullName, " -> ", policyMapper(fullName))
        print("=====Agent to Asset map=====")
        for agent in [a() for a in self.env.manager.getAgents()]:
            print(agent.getFullName(), " -> ", "{")
            for port, parent in agent.parents.items():
                print("  ", port, " : ", parent.getFullName())
            print("}")

        obs, info = self.env.reset()
        rewards = {k: 0.0 for k in obs.keys()}
        terminateds = {k: False for k in obs.keys()}
        truncateds = {k: False for k in obs.keys()}
        infos = {k: None for k in obs.keys()}
        for p in policies.values():
            p.reset()

        # シミュレーションの実行
        start_time = time.time()
        terminateds["__all__"]=False
        truncateds["__all__"]=False
        red_error = False
        blue_error = False
        error = False
        red_error_message = ''
        blue_error_message = ''
        logs = {}
        red_fighter_names = sorted([a().getFullName() for a in self.env.manager.getAgents() if 'Red' in a().getFullName()])
        blue_fighter_names = sorted([a().getFullName() for a in self.env.manager.getAgents() if 'Blue' in a().getFullName()])
        if self.match_config['make_log']:
            print('Making logs')
            logs = {'Red':{'observations': [], 'actions':[], 'infos': []}, 'Blue':{'observations': [], 'actions': [], 'infos': []}}
        time_out = self.match_config['time_out']
        step_count = 0
        while not (terminateds["__all__"] or truncateds["__all__"]):
            observation_space = self.env.get_observation_space()
            action_space = self.env.get_action_space()

            # get Red action
            red_actions = {}
            try:
                red_actions = step_foward(func = wrap_get_action_from_obs, args=(policies, rewards, terminateds, truncateds, infos, observation_space, action_space, policyMapper, obs, 'Red'), time_out=time_out)
            except Exception as e:
                print('Catched error in Red. {}'.format(e))
                red_error_message = traceback.format_exc()
                print(red_error_message)
                red_error = True

            # get Blue action
            blue_actions = {}
            try:
                blue_actions = step_foward(func = wrap_get_action_from_obs, args=(policies, rewards, terminateds, truncateds, infos, observation_space, action_space, policyMapper, obs, 'Blue'), time_out=time_out)
            except Exception as e:
                print('Catched error in Blue. {}'.format(e))
                blue_error_message = traceback.format_exc()
                print(blue_error_message)
                blue_error = True

            if red_error or blue_error:
                break
            if self.match_config['make_log']:
                self.update_logs(red_fighter_names, blue_fighter_names, logs, obs, infos, red_actions, blue_actions)
            
            actions = dict(red_actions, **blue_actions)

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

            step_count += 1

        finishtime = time.time() - start_time
        print('Time elapsed: {}[s]'.format(finishtime))

        if red_error and blue_error:
            winner = ''
            error = True
        elif red_error:
            winner = 'Blue'
            error = True
        elif blue_error:
            winner = 'Red'
            error = True
        else:
            winner = self.env.manager.getRuler()().winner
        
        detail = {
            "finishedTime": finishtime,
            "step_count": step_count,
            "numAlives":{team:float(np.sum([f.isAlive() for f in [f() for f in self.env.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()==team)]])) for team in self.env.manager.getRuler()().score},
            "endReason":self.env.manager.getRuler()().endReason.name,
            "scores": {k: v for k, v in self.env.manager.scores.items()},
            "error": error,
            "red_error_message": red_error_message,
            "blue_error_message": blue_error_message,
            "logs": logs,
            "config": self.env.manager.getManagerConfig()() # for testing
        }
        scores = {k: v for k, v in self.env.manager.scores.items()}
        detail['scores'] = scores
        print("winner=", winner)

        return winner, detail


def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


# policyMapperの定義(基本はデフォルト通り)
def policyMapper(fullName):
    agentName, modelName, policyName = fullName.split(":")
    return policyName


def wrap_get_action_from_obs(args, time_out):
    @timeout(time_out)
    def get_action_from_obs(policies, rewards, terminateds, truncateds, infos, observation_space, action_space, policyMapper, obs, team):
        actions = {}
        for k, o in obs.items():
            suff_k = k.split('_')[0]
            if policyMapper(k) in policies:
                if team in suff_k:
                    actions[k] = policies[policyMapper(k)].step(o, rewards[k], terminateds[k] or truncateds[k], infos[k], k, observation_space[k], action_space[k])
        return actions
    
    return get_action_from_obs(*args)


def step_foward(func, args, time_out):
    """
    func: decorated with timeout
    """
    try:
        r = func(args, time_out)
        return r
    except TimeoutError:
        print('TimeOut')
        return args[6].sample()


def managerConfigReplacer(config,matchInfo):
    """
    対戦カードを表すmatchInfoに従い、SimulationManagerのコンフィグを置き換える関数。
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
    """
    対戦カードを表すmatchInfoに従い、SimulationManagerのFactory側のコンフィグを置き換える関数。
    configはSimulationManager::getFactoryModelConfig(True)で得られるものであり、
    Simulationmanagerのコンストラクタに渡すenv_configのうち"Factory"キー以下の部分となる。
    """
    ret=nljson(config)
    if "Factory" in matchInfo.get("common", {}):
        ret.merge_patch(matchInfo["common"]["Factory"])
    return ret


def wrapEnvForTwoTeamCombatMatchMaking(base_class):
    """
    MatchMakerの付帯情報をgym.Envに付加するラッパー
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
