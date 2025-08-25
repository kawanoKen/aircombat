# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
import sys
import numpy as np
from ASRCAISim1.core import (
	Fighter,
	getValueFromJsonKRD,
	gravity,
	Missile,
	SimPhase,
	TeamReward,
	Track3D,
	serialize_attr_with_type_info,
	serialize_by_func,
	serialize_internal_state_by_func,
	save_with_type_info,
	load_with_type_info,
	isOutputArchive,
	isInputArchive,
)

class R7ContestPyRewardSample01(TeamReward):
	"""いくつかの観点に基づいた報酬の実装例。
	(1)Bite(誘導弾シーカで目標を捕捉)への加点
	(2)誘導弾目標のメモリトラック落ちへの減点
	(3)敵探知への加点(生存中の敵の何%を探知できているか)
	(4)過剰な機動への減点
	(5)前進・後退への更なる加減点
	(6)保持している力学的エネルギー(回転を除く)の増減による加減点
	"""
	def initialize(self):
		super().initialize()
		self.pBite=getValueFromJsonKRD(self.modelConfig,"pBite",self.randomGen,+0.0)
		self.pMemT=getValueFromJsonKRD(self.modelConfig,"pMemT",self.randomGen,+0.0)
		self.pDetect=getValueFromJsonKRD(self.modelConfig,"pDetect",self.randomGen,+0.0)
		self.pVel=getValueFromJsonKRD(self.modelConfig,"pVel",self.randomGen,+0.0)
		self.pOmega=getValueFromJsonKRD(self.modelConfig,"pOmega",self.randomGen,+0.0)
		self.pLine=getValueFromJsonKRD(self.modelConfig,"pLine",self.randomGen,+0.0)
		self.pEnergy=getValueFromJsonKRD(self.modelConfig,"pEnergy",self.randomGen,+0.0)
		self.pLineAsPeak=getValueFromJsonKRD(self.modelConfig,"pLineAsPeak",self.randomGen,False)
	def serializeInternalState(self, archive, full: bool):
		super().serializeInternalState(archive, full)
		if full:
			#observation spaceの設定
			serialize_attr_with_type_info(archive, self
				,"pBite","pMemT","pDetect","pVel","pOmega","pLine","pEnergy"
				,"pLineAsPeak"
				,"westSider","eastSider"
				,"dLine"
				,"forwardAx"
				,"numMissiles"
				,"friends","enemies"
				,"friendMsls"
			)
		serialize_attr_with_type_info(archive, self
			,"leadRange"
			,"leadRangePrev"
			,"biteFlag","memoryTrackFlag"
			,"totalEnergy"
		)
	def onEpisodeBegin(self):#初期化
		self.j_target="All"#個別のconfigによらず強制的に対象を指定する
		super().onEpisodeBegin()
		o=self.manager.getRuler()().observables
		self.westSider=o["westSider"]()
		self.eastSider=o["eastSider"]()
		self.forwardAx=o["forwardAx"]()
		self.dLine=o["dLine"]()
		crs=self.manager.getRuler()().getLocalCRS()
		self.friends={
			team:[
				f for f in [f() for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter))]
			]
			for team in self.target
		}
		self.totalEnergy={
			team:sum([
				np.linalg.norm(f.vel())**2/2+gravity*f.getHeight() for f in [f() for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter))]
			])
			for team in self.target
		}
		self.leadRangePrev={
			team:max(-self.dLine,max([
				np.dot(self.forwardAx[team],f.pos(crs)[0:2]) for f in [f() for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter))]
			]))
			for team in self.target
		}
		self.leadRange={key:value for key,value in self.leadRangePrev.items()}
		self.enemies={
			team:[
				f for f in [f() for f in self.manager.getAssets(lambda a:a.getTeam()!=team and isinstance(a,Fighter))]
			]
			for team in self.target
		}
		self.friendMsls={
			team:[
				f for f in [f() for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Missile))]
			]
			for team in self.target
		}
		self.numMissiles={team:len(self.friendMsls[team]) for team in self.target}
		self.biteFlag={team:np.full(self.numMissiles[team],False)
			for team in self.target}
		self.memoryTrackFlag={team:np.full(self.numMissiles[team],False)
			for team in self.target}
	def onInnerStepEnd(self):
		crs=self.manager.getRuler()().getLocalCRS()
		for team in self.target:
			#(1)Biteへの加点、(2)メモリトラック落ちへの減点
			for i,m in enumerate(self.friendMsls[team]):
				if(m.hasLaunched and m.isAlive):
					if(m.mode==Missile.Mode.SELF and not self.biteFlag[team][i]):
						self.reward[team]+=self.pBite
						self.biteFlag[team][i]=True
					if(m.mode==Missile.Mode.MEMORY and not self.memoryTrackFlag[team][i]):
						self.reward[team]-=self.pMemT
						self.memoryTrackFlag[team][i]=True
			#(3)敵探知への加点(生存中の敵の何%を探知できているか)(データリンク前提)
			track=[]
			for f in self.friends[team]:
				if(f.isAlive()):
					track=[Track3D(t) for t in f.observables["sensor"]["track"]]
					break
			numAlive=0
			numTracked=0
			for f in self.enemies[team]:
				if(f.isAlive()):
					numAlive+=1
					for t in track:
						if(t.isSame(f)):
							numTracked+=1
							break
			if(numAlive>0):
				self.reward[team]+=(1.0*numTracked/numAlive)*self.pDetect*self.interval[SimPhase.ON_INNERSTEP_END]*self.manager.getBaseTimeStep()
			ene=0.0
			tmp=-self.dLine
			for f in self.friends[team]:
				if(f.isAlive()):
					#(4)過剰な機動への減点(角速度ノルムに対してL2、高度方向速度に対してL1正則化)
					vAlt=f.motion.relPtoH(f.vel(),"ENU")[2]
					self.reward[team]+=(-self.pVel*abs(vAlt)-(np.linalg.norm(f.omega())**2)*self.pOmega)*self.interval[SimPhase.ON_INNERSTEP_END]*self.manager.getBaseTimeStep()
					#(5)前進・後退への更なる加減点
					tmp=max(tmp,np.dot(self.forwardAx[team],f.pos(crs)[0:2]))
				#(6)保持している力学的エネルギー(回転を除く)の増減による加減点
				ene+=np.linalg.norm(f.vel())**2/2+gravity*f.getHeight()
			self.leadRange[team]=tmp
			if(self.pLineAsPeak):
				#最高到達点で前進の加点をする場合
				if(self.leadRange[team]>self.leadRangePrev[team]):
					self.reward[team]+=(self.leadRange[team]-self.leadRangePrev[team])*self.pLine
					self.leadRangePrev[team]=self.leadRange[team]
			else:
				#都度前進・後退の加減点をする場合
				self.reward[team]+=(self.leadRange[team]-self.leadRangePrev[team])*self.pLine
				self.leadRangePrev[team]=self.leadRange[team]
			self.reward[team]+=(ene-self.totalEnergy[team])*self.pEnergy
			self.totalEnergy[team]=ene
