# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
""" 仮想シミュレータを生成するサンプル
"""
from math import *
import numpy as np
import sys
from ASRCAISim1.core import (
    Agent,
    CommunicationBuffer,
    getValueFromJsonKD,
    nljson,
    MotionState,
    PhysicalAssetAccessor,
    Quaternion,
    SimPhase,
    Track3D,
)
from BasicAgentUtility.util import (
    calcRNorm,
)

class PyVirtualSimulatorSample:
    """彼機からの仮想誘導弾を生成するサンプル
    """

    def __init__(self, config: nljson):
        """コンストラクタ。ここでは仮想シミュレータを起動しない
        """

        #各parentあたりの最大仮想誘導弾数
        self.maxNumPerParent = getValueFromJsonKD(config,"maxNumPerParent",10)

        #仮想誘導弾の生成間隔 (agent step数単位)
        self.launchInternal = getValueFromJsonKD(config,"launchInterval",5)

        #仮想誘導弾の射撃条件(彼から我へのRNormがこの値以下のとき射撃)
        self.kShoot = getValueFromJsonKD(config,"kShoot",0.5)

        self.virtualSimulator = None
        self.virtualMissiles = {}
        self.minimumDistances = {}

    def createVirtualSimulator(self, agent: Agent):
        """agentの持つ情報に基づいて仮想シミュレータを起動する。agentのvalidate()で呼び出すことを想定。
        """
        option = nljson({
            "patch": {
                "Manager": {
                    "TimeStep": {
                        "defaultAgentStepInterval": agent.interval[SimPhase.AGENT_STEP]
                    },
                    "skipNoAgentStep": False
                }
            }
        })
        self.virtualSimulator = agent.manager.createVirtualSimulationManager(option)
        self.virtualSimulator.reset()

        self.virtualMissiles = {}
        self.minimumDistances = {}

    def step(self, agent: Agent, tracks: list[Track3D]):
        """agentの持つ情報に基づいて仮想シミュレータの時間を進め、情報を取得する。仮想敵の情報をtracksで与える。
            agentのdeploy(action)で毎回呼び出すことを想定。
        """
        now = agent.manager.getTickCount()
        for port, parent in agent.parents.items():
            fullName = parent.getFullName()

            if parent.isAlive():
                motion = MotionState(parent.observables.at("motion"))
                crs = agent.manager.getRootCRS()
                time = agent.manager.getTime()
                selfTrack = Track3D(
                    parent.getUUIDForTrack(),
                    crs,
                    time,
                    motion.pos(crs),
                    motion.vel(crs)
                )
                
                if not fullName in self.virtualMissiles:
                        self.virtualMissiles[fullName] = []
                if not fullName in self.minimumDistances:
                        self.minimumDistances[fullName] = []

                #仮想誘導弾の諸元を更新
                self.minimumDistances[fullName]=np.inf
                hit = False
                i=0
                while i < len(self.virtualMissiles[fullName]):
                    missile = self.virtualMissiles[fullName][i]
                    if missile.isAlive():
                        missile.communicationBuffers["MissileComm:"+missile.getFullName()]().send({
                            "target": selfTrack
                        },CommunicationBuffer.UpdateType.MERGE)

                        dt = agent.interval[SimPhase.AGENT_STEP]*agent.manager.getBaseTimeStep()

                        if missile.hitCheck(motion.pos(missile.getParentCRS()),motion.pos(missile.getParentCRS())-motion.vel(missile.getParentCRS())*dt):
                            # 命中
                            missile.manager.requestToKillAsset(missile)
                            hit=True

                        self.minimumDistances[fullName] = min(
                            self.minimumDistances[fullName],
                            np.linalg.norm(missile.pos()-motion.pos(missile.getParentCRS()))
                        )
                        i+=1
                    else:
                        # 飛翔終了していたら消す
                        self.virtualMissiles[fullName].pop(i)

                    #print(
                    #    fullName," tick=",agent.manager.getTickCount(),
                    #    ": virtualMissiles size=",len(self.virtualMissiles[fullName]),
                    #    ", minimumDistance=",self.minimumDistances[fullName],
                    #    ", hit=",hit
                    #)
            else:
                # parentが消えたら仮想誘導弾も消す
                if fullName in self.virtualMissiles:
                    for missile in self.virtualMissiles[fullName]:
                        missile.manager.requestToKillAsset(missile)

                    self.virtualMissiles.pop(fullName)
                    self.minimumDistances.pop(fullName)

        # 仮想シミュレータの時間を進める
        while self.virtualSimulator.getTickCount() < now:
            #print("call virtual simulator's step tick=",now," (v ",self.virtualSimulator.getTickCount(),")")
            self.virtualSimulator.step({})

        # 生存中のparentsに対して、新たな仮想誘導弾を生成する
        for port, parent in agent.parents.items():
            fullName = parent.getFullName()
            if parent.isAlive():
                motion = MotionState(parent.observables.at("motion"))
                crs = agent.manager.getRootCRS()
                time = agent.manager.getTime()
                selfTrack = Track3D(
                    parent.getUUIDForTrack(),
                    crs,
                    time,
                    motion.pos(crs),
                    motion.vel(crs)
                )
                for t in tracks:
                    if len(self.virtualMissiles[fullName]) < self.maxNumPerParent:
                        shooterMotion = MotionState(
                            crs,
                            time,
                            t.pos(crs),
                            t.vel(crs),
                            np.array([0,0,0]),
                            Quaternion(1,0,0,0),
                            "FSD"
                        )
                        if self.launchCondition(parent, shooterMotion,selfTrack):
                            launchCommand=nljson({
                                "target": selfTrack,
                                "motion": shooterMotion
                            })
                            self.virtualMissiles[fullName].append(parent.launchInVirtual(self.virtualSimulator,launchCommand))

    def launchCondition(self, parent: PhysicalAssetAccessor, motion: MotionState, track: Track3D) -> bool:
        """仮想誘導弾の射撃条件を指定する。
            この例では、motionからtrackへのRNormがkShoot以下の場合に射撃する。
        """
        return calcRNorm(parent, motion, track, False) <= self.kShoot
