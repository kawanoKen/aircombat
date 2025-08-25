# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
import sys,os,time
import numpy as np
import datetime
from ASRCAISim1.core import (
	Callback,
	getValueFromJsonKRD,
	Fighter,
	Missile,
	serialize_attr_with_type_info,
	isInputArchive,
)

class EpisodewiseLoggerBase(Callback):
	"""エピソード単位でログファイルを生成するためのベースクラス。
	エピソード間に跨るログファイルを生成したい場合は、ファイルの生成タイミングを工夫すれば同様に実装できる。
	"""
	def initialize(self):
		super().initialize()
		self.prefix=getValueFromJsonKRD(self.modelConfig,"prefix",self.randomGen,"")
		self.outerInterval=getValueFromJsonKRD(self.modelConfig,"outerInterval",self.randomGen,1)
		self.innerInterval=getValueFromJsonKRD(self.modelConfig,"innerInterval",self.randomGen,1)
		self.outerCounter=0
		self.innerCounter=0
		self.isFileCreated=False
		self.file=None
		self.fileAbsPath=""
		self.crsConfig=None
		if "crs" in self.instanceConfig:
			self.crsConfig=self.instanceConfig["crs"]
		elif "crs" in self.modelConfig:
			self.crsConfig=self.modelConfig["crs"]
		self.crs=None
	def serializeInternalState(self, archive, full: bool):
		super().serializeInternalState(archive, full)
		if full:
			serialize_attr_with_type_info(archive, self
				,"prefix"
				,"outerInterval"
				,"innerInterval"
				,"crsConfig"
				,"crs"
			)
		serialize_attr_with_type_info(archive, self
			,"outerCounter"
			,"innerCounter"
			,"isFileCreated"
		)
		if self.isFileCreated:
			serialize_attr_with_type_info(archive, self, "fileAbsPath")
			if isInputArchive(archive):
				self.file=open(self.fileAbsPath,'a')
		else:
			if isInputArchive(archive):
				self.fileAbsPath=""
				self.file=None
	def onEpisodeBegin(self):
		if self.crsConfig is None:
			self.crs = self.manager.getRootCRS()
		else:
			self.crs = self.manager.createOrGetEntity(self.crsConfig)
		self.fileAbsPath=os.path.abspath(
			self.prefix+"_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".csv"
		)
		os.makedirs(os.path.dirname(self.fileAbsPath),exist_ok=True)
		self.file=open(self.fileAbsPath,'w')
		self.isFileCreated=True
		self.makeHeader()
		self.makeFrameOnStep()
	def onStepEnd(self):
		if(self.outerCounter%self.outerInterval==0):
			self.makeFrameOnStep()
		self.outerCounter+=1
	def onInnerStepEnd(self):
		if(self.innerCounter%self.innerInterval==0):
			self.makeFrameOnInnerStep()
		self.innerCounter+=1
	def onEpisodeEnd(self):
		self.file.close()
		self.file=None
		self.isFileCreated=False
	def makeHeader(self):
		pass
	def makeFrameOnStep(self):
		pass
	def makeFrameOnInnerStep(self):
		pass

class BasicLogger(EpisodewiseLoggerBase):
	"""エピソード単位のログ出力の実装例。
	各陣営の戦闘機と誘導弾の基本的な情報を吐き出す。
	"""
	def makeHeader(self):
		row=["Time[s]"]
		teams=self.manager.getTeams()
		for team in teams:
			row.extend([team+".remFighters",team+".remMissiles"])
		for team in teams:
			for f in self.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()==team):
				name=f().getFullName()
				row.extend([name+".isAlive"])
				row.extend([name+".pos.x[m]",name+".pos.y[m]",name+".pos.z[m]"])
				row.extend([name+".vel.x[m/s]",name+".vel.y[m/s]",name+".vel.z[m/s]"])
				row.extend([name+".att.roll[rad]",name+".att.pitch[rad]",name+".att.yaw[rad]"])
				row.extend([name+".omega.x[rad/s]",name+".omega.y[rad/s]",name+".omega.z[rad/s]"])
		for team in teams:
			for f in self.manager.getAssets(lambda a:isinstance(a,Missile) and a.getTeam()==team):
				name=f().getFullName()
				row.extend([name+".isFlying",name+".target.truth"])
				row.extend([name+".pos.x[m]",name+".pos.y[m]",name+".pos.z[m]"])
				row.extend([name+".vel.x[m/s]",name+".vel.y[m/s]",name+".vel.z[m/s]"])
				row.extend([name+".att.roll[rad]",name+".att.pitch[rad]",name+".att.yaw[rad]"])
				row.extend([name+".omega.x[rad/s]",name+".omega.y[rad/s]",name+".omega.z[rad/s]"])
		self.file.write(','.join(row)+"\n")
	def makeFrameOnStep(self):
		pass
	def makeFrameOnInnerStep(self):
		row=[]
		row.append(format(self.manager.getElapsedTime(),'0.6f'))
		teams=self.manager.getTeams()
		for team in teams:
			remFighters=0
			remMissiles=0
			for f in self.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()==team):
				f=f()
				if(f.isAlive()):
					remFighters+=1
			for f in self.manager.getAssets(lambda a:isinstance(a,Missile) and a.getTeam()==team):
				f=f()
				if(f.isAlive()):
					remMissiles+=1
			row.extend([str(remFighters),str(remMissiles)])
		for team in teams:
			for f in self.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()==team):
				f=f()
				row.extend([str(f.isAlive())])
				if(f.isAlive()):
					row.extend([format(x,'+0.16e') for x in f.pos(self.crs)])
					row.extend([format(x,'+0.16e') for x in f.vel(self.crs)])
					row.extend([format(x,'+0.16e') for x in f.toEulerAngle()])
					row.extend([format(x,'+0.16e') for x in f.omega(self.crs)])
				else:
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
		for team in teams:
			for f in self.manager.getAssets(lambda a:isinstance(a,Missile) and a.getTeam()==team):
				f=f()
				if(f.isAlive() and f.hasLaunched):
					tgtName="None"
					for tgt in self.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()!=team):
						tgt=tgt()
						if(f.target.isSame(tgt)):
							tgtName=tgt.getFullName()
							break
					row.extend(["True",tgtName])
					row.extend([format(x,'+0.16e') for x in f.pos(self.crs)])
					row.extend([format(x,'+0.16e') for x in f.vel(self.crs)])
					roll=0.0
					ex=f.vel(self.crs)/np.linalg.norm(f.vel(self.crs))
					yaw=atan2(ex[1],ex[0])
					pitch=atan2(-ex[2],sqrt(ex[0]*ex[0]+ex[1]*ex[1]))
					row.extend([format(x,'+0.16e') for x in [roll,pitch,yaw]])
					row.extend([format(x,'+0.16e') for x in f.omega(self.crs)])
				else:
					row.extend(["False","None"])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
					row.extend([format(x,'+0.16e') for x in np.array([0.,0.,0.])])
		self.file.write(','.join(row)+"\n")
