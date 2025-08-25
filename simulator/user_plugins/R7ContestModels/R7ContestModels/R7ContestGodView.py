# Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from ASRCAISim1.utility.GraphicUtility import *
from ASRCAISim1.core import *
from ASRCAISim1.viewer.GodView import GodView,GodViewPanel

class R7ContestGodView(GodView):
    """R7年度コンテスト用のGodView
    """
    defaultPanelConfigPath = os.path.join(os.path.dirname(__file__),"config/ui","R7ContestGodViewUIConfig.json")
    def makePanel(self):
        """画面描画を行うPanelオブジェクトを返す。
        """
        return R7ContestGodViewPanel(self.panelConfig)

class R7ContestGodViewPanel(GodViewPanel):
    """R7年度コンテスト用のGodViewPanel
    """
    def identifyFighterModelToDraw(self,f):
        """ 描画用の機体種別を識別する。
        """
        if f["observables"]["spec"]["weapon"]["numMsls"] > 0: #Fighter
            return "Default"
        else: #Escorted
            return "Escorted"
    def getPolicyNameToDisplay(self, frame):
        """各陣営のAgentを表す名称を取得するもの。
            R7年度コンテストでは護衛対象機も独立したAgentを持っているため、それを除いて先頭に来るもののpolicy名を用いる。

        Args:
            frame (dict): 現在のフレームデータ

        Returns:
            str: Agent名表示用の文字列
        """
        ruler=frame["ruler"]
        policyName = {}
        for team in [ruler["westSider"], ruler["eastSider"]]:
            policyName[team]=""
            for agentFullName,agent in frame["agents"].items():
                if agent["team"]==team:
                    name,model,policy=agentFullName.split(":")
                    if not model.endswith("EscortedAgent"):
                        if policy.startswith("Policy_"):
                            policyName[team]=policy[7:]
                        else:
                            policyName[team]=policy
                        break
        return policyName
    def getRewardStringToDisplay(self, team, frame):
        """報酬を表示する際の文字列を取得するもの。
            R7年度コンテストでは護衛対象機の報酬を表示対象から除外する。
        Args:
            team str: 計算対象の陣営
            frame dict: 現在のフレームデータ

        Returns:
            str: 報酬表示用の文字列
        """
        def stringifyTotalRewards(tr):
            if isinstance(tr,float):
                return "{:.1f}".format(tr)
            else:
                if len(tr)==1:
                    return "{:.1f}".format(next(iter(tr.values())))
                else:
                    return "{"+",".join([k+":{:.1f}".format(v) for k,v in tr.items()])+"}"
        def check(agentFullName):
            name,model,policy=agentFullName.split(":")
            return not model.endswith("EscortedAgent")
        reward=[stringifyTotalRewards(frame["totalRewards"][agentFullName]) for agentFullName,agent in frame["agents"].items() if agent["team"]==team and check(agentFullName)]
        return str(reward)
