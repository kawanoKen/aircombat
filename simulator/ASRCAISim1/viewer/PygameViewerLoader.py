# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import os
import copy
import json
import pygame
from pygame.locals import QUIT
import cv2
import numpy as np
from ASRCAISim1.utility.GraphicUtility import *
from ASRCAISim1.core import (
	EntityIdentifier,
	getValueFromJsonKD,
	getValueFromJsonKRD,
	MotionState,
	SimulationManager,
	nljson,
	std_mt19937,
)
import pickle
import bz2
import glob

class PygameViewerLoader:
	"""PygameViewerのGUI表示を、GUIStateLoggerにより保存されたログを読み込んで行いつつ、
	連番画像又は動画として保存するためのクラス。(保存の有無は選択可能。)
	インターフェースはCallbackに準じているが、SimulationManagerとは独立に動くものである。
	PygameViewerクラスと同様に、実際の描画処理を行うPanelオブジェクトをmakePanelメソッドのオーバーライドにより指定する。
	<使い方の例>
	```python
	loader=GodViewLoader({
		"globPattern":"~/logs/hoge_*.dat",
		"outputPrefix":"~/movies/movie",
		"asVideo":True,
		"fps":60
	})
	loader.run()
	```
	"""
	defaultPanelConfigPath = None
	def __init__(self,modelConfig={},instanceConfig={}):
		# 乱数の再現
		self.randomGen=std_mt19937()
		if "seed" in instanceConfig:
			self.randomGen.seed(instanceConfig["seed"])
		firstTicks={}
		if "firstTick" in instanceConfig:
			firstTicks=instanceConfig["firstTick"]
		elif "firstTick" in modelConfig:
			firstTicks=modelConfig["firstTick"]
		firstTickUnit=firstTicks.get("unit", "tick")
		if firstTickUnit == "time":
			getValueFromJsonKRD(firstTicks,"value",self.randomGen,0.0)
		elif firstTickUnit == "tick":
			getValueFromJsonKRD(firstTicks,"value",self.randomGen,0)
		intervals={}
		if "interval" in instanceConfig:
			intervals=instanceConfig["interval"]
		elif "interval" in modelConfig:
			intervals=modelConfig["interval"]
		intervalUnit=intervals.get("unit", "tick")
		if intervalUnit == "time":
			getValueFromJsonKRD(intervals,"value",self.randomGen,0.1)
		elif intervalUnit == "tick":
			getValueFromJsonKRD(intervals,"value",self.randomGen,1)
		# 乱数の再現終わり
		self.modelConfig=nljson(modelConfig)
		self.instanceConfig=nljson(instanceConfig)
		self.globPattern=getValueFromJsonKD(self.modelConfig,"globPattern","") #読み込み対象のファイルをglobパターンで指定する。		
		self.outputDir=getValueFromJsonKD(self.modelConfig,"outputDir",None) #保存先のprefixを指定する。未指定の場合は保存せず表示のみとなる。
		self.outputFileNamePrefix=getValueFromJsonKD(self.modelConfig,"outputFileNamePrefix",None) #保存先のprefixを指定する。未指定の場合は保存せず表示のみとなる。
		self.dataPaths=sorted(glob.glob(self.globPattern,recursive=True))
		self.asVideo=getValueFromJsonKD(self.modelConfig,"asVideo",True) #動画として保存するか連番画像として保存するか。
		self.skipRate=getValueFromJsonKD(self.modelConfig,"skipRate",1) #保存するフレームの間隔
		self.fps=getValueFromJsonKD(self.modelConfig,"fps",10) #動画として保存する際のフレームレート
		print("self.dataPaths=",self.dataPaths)
		self.episodeCounter=0
		self.width=getValueFromJsonKRD(self.modelConfig,"width",self.randomGen,1280)
		self.height=getValueFromJsonKRD(self.modelConfig,"height",self.randomGen,720)
		self.initial_position = getValueFromJsonKD(self.modelConfig,"initial_position",None)
		if self.defaultPanelConfigPath is not None:
			with open(self.defaultPanelConfigPath,'r') as f:
				self.panelConfig = nljson(json.load(f))
		else:
			self.panelConfig = None
		for panelConfig in [self.modelConfig.get("panelConfig", None), self.instanceConfig.get("panelConfig", None)]:
			if isinstance(panelConfig, nljson):
				if panelConfig.is_object():
					pass
				elif panelConfig.is_string():
					with open(panelConfig(),'r') as f:
						panelConfig = json.load(f)
				elif panelConfig.is_null():
					pass
				else:
					raise ValueError(f"panelConfig must be dict, str or None. Given {panelConfig.type()}")
			elif isinstance(panelConfig, dict):
				pass
			elif isinstance(panelConfig, str):
				with open(panelConfig,'r') as f:
					panelConfig = json.load(f)
			elif panelConfig is None:
				pass
			else:
				raise ValueError(f"panelConfig must be dict, str or None. Given {type(panelConfig)}")
			if self.panelConfig is not None:
				if panelConfig is not None:
					self.panelConfig.merge_patch(panelConfig)
			else:
				if isinstance(panelConfig, nljson):
					self.panelConfig = panelConfig()
				else:
					self.panelConfig = nljson(panelConfig)
		self.panelConfig = self.panelConfig()
		self.panel=None
		self.isValid=False
		self.manualDone=False
		self.unpickler=None
		self.manager=None

	def validate(self):
		self.isValid=True
		if self.initial_position is not None:
			import os
			old = os.getenv('SDL_VIDEO_WINDOW_POS',None)
			os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int(self.initial_position[0])},{int(self.initial_position[1])}"
			if old is not None:
				os.environ['SDL_VIDEO_WINDOW_POS'] = old
		pygame.init()
		self.font=pygame.font.Font('freesansbold.ttf',12)
		self.window=pygame.display.set_mode(
			(self.width,self.height),pygame.DOUBLEBUF | pygame.OPENGL | pygame.OPENGLBLIT)
		self.clock=pygame.time.Clock()
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		self.panel=self.makePanel()

	def close(self):
		pygame.quit()

	def run(self):
		if(not self.isValid):
			self.validate()
		self.episodeCounter=0
		for path in self.dataPaths:
			self.episodeCounter += 1
			self.currentFileName=os.path.splitext(os.path.basename(path))[0]
			print("current: ",self.currentFileName)
			with bz2.BZ2File(path,"rb") as fin:
				unpickler=pickle.Unpickler(fin)
				self.header=unpickler.load()
				if(isinstance(self.header,nljson)):
					self.header=self.header()
				self.currentFrame=unpickler.load()
				if(isinstance(self.currentFrame,nljson)):
					self.currentFrame=self.currentFrame()
				self.onEpisodeBegin()
				while True:
					try:
						self.currentFrame=unpickler.load()
						if(isinstance(self.currentFrame,nljson)):
							self.currentFrame=self.currentFrame()
						self.onInnerStepBegin()
						self.onInnerStepEnd()
					except EOFError:
						break
				self.onEpisodeEnd()
				del unpickler

	def saveImage(self):
		try:
			if(not self.asVideo):
				path=os.path.join(
					self.outputDir,
					self.outputFileNamePrefix+"e{:04d}_f{:06.1f}.png".format(self.episodeCounter,self.frameCounter)
				)
				os.makedirs(os.path.dirname(path),exist_ok=True)
			width=self.width
			height=self.height
			img=np.zeros((height,width,3),dtype=np.uint8)
			glReadPixels(0,0,width,height,GL_BGR,GL_UNSIGNED_BYTE,img.data)
			img=np.ascontiguousarray(img[::-1])
			if(self.asVideo):
				self.video.write(img)
			else:
				cv2.imwrite(path,img)
		except Exception as e:
			print("failed to save an image:")
			raise e

	def createLocalSimulationManager(self):
		localConfig={
			"Manager":{
				"uuid":self.header["uuid"],
				"episode_index":self.header["episode_index"],
				"Epoch":{
					k: v for k, v  in list(self.header["epoch"].items())+list(self.header["epoch_deltas"].items())
				},
				"CoordinateReferenceSystem":{
					"preset": {k: v["config"] for k,v in self.header["coordinateReferenceSystem"]["preset"].items()},
					"root": self.header["coordinateReferenceSystem"]["root"],
				},
				"Viewer": "None",
				"Ruler":{
					"class":"Ruler",
					"modelConfig":{},
					"instanceConfig":{
						"crs":self.currentFrame["ruler"]["localCRS"]
					}
				},
				"AssetConfigDispatcher":{},
				"AgentConfigDispatcher":{},
			},
			"Factory":self.header["factory"]
		}
		self.manager=SimulationManager(
			localConfig,
			self.header["worker_index"],
			self.header["vector_index"],
		)
		self.manager.configure()
		self.manager.loadInternalStateOfEntity(
			[(EntityIdentifier(i),b) for i,b in self.header["coordinateReferenceSystem"]["internalStateIndices"]],
			self.header["coordinateReferenceSystem"]["internalStates"]
		)
		self.manager.reconstructEntities(self.currentFrame["crses"]["reconstructionInfos"])
		self.manager.loadInternalStateOfEntity(
			[(EntityIdentifier(i),b) for i,b in self.currentFrame["crses"]["internalStateIndices"]],
			self.currentFrame["crses"]["internalStates"]
		)

	def onEpisodeBegin(self):
		"""エピソードの開始時(reset関数の最後)に呼ばれる。
		"""
		if(not self.isValid):
			self.validate()

		self.frameCounter=0

		if(self.asVideo):
			fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
			path=os.path.join(
				self.outputDir,
				self.outputFileNamePrefix+"{}.mp4".format(self.currentFileName)
			)
			os.makedirs(os.path.dirname(path),exist_ok=True)
			self.video=cv2.VideoWriter(
				path,fourcc,self.fps,(self.width,self.height))

		if self.manager is not None:
			self.manager.purge()
			self.manager=None

		self.createLocalSimulationManager()
		if hasattr(self.panel, "onEpisodeBegin"):
			self.panel.onEpisodeBegin(self.manager.getViewer(),self.header,self.currentFrame)

	def onStepBegin(self):
		"""gym.Envとしてのstepの開始時に呼ばれる。
		"""
		pass

	def onStepEnd(self):
		"""gym.Envとしてのstepの最後に呼ばれる。
		Managerは得点計算⇛報酬計算⇛その他コールバック⇛画面表示の順で呼ぶ。
		"""
		pass

	def onInnerStepBegin(self):
		"""インナーループの各ステップの開始時(controlの前)に呼ばれる。
		"""
		for e in pygame.event.get():
			if(e.type==QUIT):
				self.manualDone=True

	def onInnerStepEnd(self):
		"""#インナーループの各ステップの最後(perceiveの後)に呼ばれる。
		"""
		if(self.isValid):
			self.frameCounter+=1
			self.manager.reconstructEntities(self.currentFrame["crses"]["reconstructionInfos"])
			self.manager.loadInternalStateOfEntity(
				[(EntityIdentifier(i),b) for i,b in self.currentFrame["crses"]["internalStateIndices"]],
				self.currentFrame["crses"]["internalStates"]
			)
			self.manager.setInternalMotionStatesOfAffineCRSes(
				[(EntityIdentifier(i),MotionState(m)) for i,m in self.currentFrame["crses"]["internalMotionStates"]]
			)
			if (self.frameCounter-1)%self.skipRate==0:
				self.display(self.currentFrame)
				if(self.outputFileNamePrefix is not None):
					self.saveImage()

	def onEpisodeEnd(self):
		"""エピソードの終了時(step関数の最後でdone==Trueの場合)に呼ばれる
		"""
		if(self.asVideo):
			self.video.release()
		if(self.manualDone):
			pygame.quit()

	def display(self,frame):
		"""画面描画を行う。実際の描画処理はPanelに投げる。
		"""
		self.panel.display(frame)

	def makePanel(self):
		"""画面描画を行うPanelオブジェクトを返す。派生クラスでオーバーライドする。
		"""
		raise NotImplementedError
