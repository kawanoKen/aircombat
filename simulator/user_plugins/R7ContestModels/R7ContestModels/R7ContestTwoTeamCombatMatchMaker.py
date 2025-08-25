# Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os
import time
import datetime
import numpy as np
import gymnasium as gym
from collections import defaultdict
import copy
import cloudpickle
from ASRCAISim1.plugins.rayUtility.utility.TransparentRemoteObject import unwrap_if_needed
from ASRCAISim1.plugins.MatchMaker.TwoTeamCombatMatchMaker import TwoTeamCombatMatchMaker, getRandomizedInitialState, invert

def getForEscort(phase):
    #護衛対象機の初期位置を計算する
    p1=np.array([60000.0,100000.0,-10000.0])
    p2=np.array([-60000.0,100000.0,-10000.0])
    dp=p2-p1
    radius=5000.0
    distance=np.sqrt(dp[0]**2+dp[1]**2)
    ex=dp/np.linalg.norm(dp)
    ey=np.cross(np.array([0.,0.,1.]),ex)
    ey/=np.linalg.norm(ey)

    length=2*(radius*np.pi+distance)
    phase2rad=length/radius
    anchors=[
        (radius*np.pi/2.)/length,
        (radius*np.pi/2.+distance)/length,
        (radius*np.pi*3/2.+distance)/length,
        (radius*np.pi*3/2.+2*distance)/length,
    ]
    if phase < anchors[0]:
        angle=phase*phase2rad
        pos=p1+radius*(-ex*np.cos(angle)-ey*np.sin(angle))
        heading=angle/np.pi*180
    elif phase < anchors[1]:
        pos=p1+(ex*(phase-anchors[0])*length-ey*radius)
        heading=180.0
    elif phase < anchors[2]:
        angle=(phase-anchors[1])*phase2rad
        pos=p2+radius*(ex*np.sin(angle)-ey*np.cos(angle))
        heading=90.0+angle/np.pi*180
    elif phase < anchors[3]:
        pos=p2+(-ex*(phase-anchors[2])*length+ey*radius)
        heading=0.0
    else:
        angle=(1.0-phase)*phase2rad
        pos=p1+radius*(-ex*np.cos(angle)+ey*np.sin(angle))
        heading=-angle/np.pi*180
    return {
        "pos":pos,
        "vel":300.0,
        "heading": heading,
    }

class R7ContestTwoTeamCombatMatchMaker(TwoTeamCombatMatchMaker):
    def initialize(self,config):
        super().initialize(config)
        self.youth=self.match_config.get("youth",False)
        self.randomized_asset_spec=self.match_config.get("randomized_asset_spec",{})
        if self.randomized_asset_spec is None:
            self.randomized_asset_spec={}
        self.symmetric_randomization=self.match_config.get("symmetric_randomization",True)
        self.shuffle_fighter_order=self.match_config.get("shuffle_fighter_order",False)
        self.heterogeneous_randomization=self.match_config.get("heterogeneous_randomization",False)
    def makeNextMatch(self,matchType,worker_index):
        """対戦カードを生成する。Agent、Assetの初期配置、Assetの性能の3種類の設定を生成する。
        """
        #return super().makeNextMatch(matchType,worker_index)
        ret={
            "teams": {
                self.teams[0]:{},
                self.teams[1]:{},
            },
            "common": {
            }
        }
        ret["teams"]=self.setAgent(ret["teams"],matchType,worker_index)
        ret["teams"]=self.setInitialState(ret["teams"],matchType,worker_index)
        ret["common"]=self.setCommonCondition(ret["common"],matchType,worker_index)
        return ret

    def setAgent(self,ret,matchType,worker_index):
        """Agentの設定を行う。このサンプルでは親クラスと同様に、以下のように設定される。
        第一陣営(Blue)側：学習中の"Learner"又は一定確率(expert_ratio)で教師役の"Expert"
        第二陣営(Red)側：一定エピソード数(warm_up_episodes)の経過前は常に初期行動判断モデル("Initial")とし、
            経過後は(1)学習中重みの直近のコピー、(2)初期行動判断モデル("Initial")、(3)過去の保存された重みから一様分布で選択
            の3種類を4:4:2の割合で選択(ただし、(3)に該当する重みが存在しない場合は(1)とする。)
        expert_ratioとwarm_up_episodesはmatch_configで指定する。
        また、matchType,worker_indexは特に使用しない。
        """
        ret=super().setAgent(ret,matchType,worker_index)
        return ret

    def setInitialState(self,ret,matchType,worker_index):
        """各Assetの初期配置の設定を行う。
        このサンプルでは、pos,vel,headingの3種類の値を設定する。
        第一陣営(Blue)側の戦闘機は、match_configに"initial_state_lower"及び"initial_state_upper"として与えた上下限の範囲内でランダムに配置する。
        第二陣営(Red)側の戦闘機は、Blue側と点対称に配置する。
        護衛対象機は対称性を持たせず、周回軌道上のランダムな位相に配置する。
        """
        if "initial_state_lower" in self.match_config and "initial_state_upper" in self.match_config:
            lower_bound=self.match_config["initial_state_lower"]
            upper_bound=self.match_config["initial_state_upper"]
            if self.youth:
                lower_bound["pos"][2]=-10000.0
                upper_bound["pos"][2]=-10000.0
            initialStates=getRandomizedInitialState(
                self.teams,
                lower_bound,
                upper_bound,
                self.match_config.get("initial_state_number",5)-1,
                self.match_config.get("symmetric_initial_state",True)
            )
            for team in self.teams:
                ret[team]["InitialState"]=initialStates[team]["InitialState"]
    
            ret[self.teams[0]]["InitialState"].append(getForEscort(np.random.rand()))
            ret[self.teams[1]]["InitialState"].append(invert(getForEscort(np.random.rand())))
        return ret

    def setCommonCondition(self,ret,matchType,worker_index):
        """teamに依存しない条件(Rule等)を記述する。
        """
        ret=super().setCommonCondition(ret,matchType,worker_index)
        ret=self.setFactoryModel(ret,matchType,worker_index)
        return ret

    def setFactoryModel(self,ret,matchType,worker_index):
        """各Assetの性能に関する設定を行う。
        """
        division = "Youth" if self.youth else "Open"
        number=self.match_config.get("initial_state_number",5)-1
        fac={
            "Controller":{
            },
            "PhysicalAsset":{
            }
        }

        table=self.randomized_asset_spec
        def getModelConfigs():
            humanIntervention={
                "model":"R7ContestModels.R7ContestHumanIntervention",
                "config":{
                    "delay": 3.0,
                }
            }
            fighter={
                "model":"R7ContestModels.R7Contest"+division+"Fighter",
                "config":{
                    "pilot":{
                        "model":"R7ContestHumanIntervention"
                    },
                    "sensor":{
                        "radar":"R7Contest"+division+"AircraftRadar"
                    },
                    "weapon":{
                        "missile":"R7Contest"+division+"Missile",
                        "numMsls":6
                    },
                    "stealth":{
                        "rcsScale":1.0,
                    },
                    "dynamics":{
                        "scale":{
                            "vMin": 1.0,
                            "vMax": 1.0,
                        }
                    }
                }
            }
            missile={
                "model":"R7ContestModels.R7Contest"+division+"Missile",
                "config":{
                    "thrust": 30017.9989,
                }
            }
            radar={
                "model":"R7ContestModels.R7Contest"+division+"AircraftRadar",
                "config":{
                    "Lref": 100000000.0 if division=="Youth" else 100000.0,
                    "thetaFOR": 180.0 if division=="Youth" else 90.0
                }
            }
            if "radar_range" in table or "radar_coverage" in table:
                if "radar_range" in table:
                    radar["config"]["Lref"]*=np.random.uniform(table["radar_range"][0],table["radar_range"][1])
                if "radar_coverage" in table:
                    radar["config"]["thetaFOR"]=np.random.uniform(table["radar_coverage"][0],table["radar_coverage"][1])
            if "rcs" in table or "maximum_speed" in table or "num_missiles" in table:
                if "rcs" in table:
                    fighter["config"]["stealth"]["rcsScale"]*=10.0**np.random.uniform(np.log10(table["rcs_scale"][0]),np.log10(table["rcs_scale"][1]))
                if "maximum_speed" in table:
                    scale=np.random.uniform(table["maximum_speed"][0],table["maximum_speed"][1])
                    fighter["config"]["dynamics"]["scale"]["vMax"]=scale
                    if scale<1:
                        fighter["config"]["dynamics"]["scale"]["vMin"]=scale
                if "num_missiles" in table:
                    fighter["config"]["weapon"]["numMsls"]=np.random.randint(table["num_missiles"][0],table["num_missiles"][1])
            if "missile_thrust" in table:
                missile["config"]["thrust"]*=np.random.uniform(table["missile_thrust"][0],table["missile_thrust"][1])
            if "shot_approval_delay" in table:
                humanIntervention["config"]["delay"]=np.random.randint(table["shot_approval_delay"][0],table["shot_approval_delay"][1]+1)
            return humanIntervention, radar, fighter, missile

        order = np.random.permutation(range(number)) if self.shuffle_fighter_order else np.array(range(number))
        humanIntervention, radar, fighter, missile = None, None, None, None
        for i in range(number):
            if self.heterogeneous_randomization or humanIntervention is None:
                humanIntervention, radar, fighter, missile = getModelConfigs()
            for tIdx, team in enumerate(self.teams):
                if tIdx>0 and not self.symmetric_randomization:
                    humanIntervention, radar, fighter, missile = getModelConfigs()
                index = str(i+1) if tIdx==0 else str(order[i]+1)
                fac["Controller"][team+"HumanIntervention"+index]=copy.deepcopy(humanIntervention)
                fac["PhysicalAsset"][team+"AircraftRadar"+index]=copy.deepcopy(radar)
                fac["PhysicalAsset"][team+"Fighter"+index]=copy.deepcopy(fighter)
                fac["PhysicalAsset"][team+"Missile"+index]=copy.deepcopy(missile)
        for i in range(number):
            for tIdx, team in enumerate(self.teams):
                fac["PhysicalAsset"][team+"Fighter"+str(i+1)]["config"]["pilot"]["model"]=team+"HumanIntervention"+str(i+1)
                fac["PhysicalAsset"][team+"Fighter"+str(i+1)]["config"]["weapon"]["missile"]=team+"Missile"+str(i+1)
                fac["PhysicalAsset"][team+"Fighter"+str(i+1)]["config"]["sensor"]["radar"]=team+"AircraftRadar"+str(i+1)
        ret["Factory"]=fac
        return ret
