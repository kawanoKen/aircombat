# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import os
import sys
import copy
import json
import pygame
from pygame.locals import QUIT
from ASRCAISim1.core import (
	nljson,
	Viewer,
	getValueFromJsonKD,
	getValueFromJsonKRD,
	serialize_attr_with_type_info,
	serialize_internal_state_of_attr,
	serialize_internal_state_by_func,
	isInputArchive,
)
from ASRCAISim1.utility.GraphicUtility import *

class PygameViewer(Viewer):
	"""pygameを用いて、戦況を描画するための基底クラス。
	"""
	defaultPanelConfigPath = None
	def initialize(self):
		super().initialize()
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
		self.creator=None

	def serializeInternalState(self, archive, full: bool):
		super().serializeInternalState(archive, full)
		if full:
			serialize_attr_with_type_info(archive, self
				, "width"
				, "height"
				, "panelConfig"
				, "initial_position"
			)
			if self.isValid:
				serialize_attr_with_type_info(archive, self, "fps")
				if isInputArchive(archive):
					if self.initial_position is not None:
						import os
						old = os.getenv('SDL_VIDEO_WINDOW_POS',None)
						os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int(self.initial_position[0])},{int(self.initial_position[1])}"
						if old is not None:
							os.environ['SDL_VIDEO_WINDOW_POS'] = old
					pygame.init()
					self.window=pygame.display.set_mode((self.width,self.height),pygame.DOUBLEBUF|pygame.OPENGL|pygame.OPENGLBLIT)
					self.clock=pygame.time.Clock()
					glEnable(GL_DEPTH_TEST)
					glEnable(GL_BLEND)
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
					self.creator=self.getDataFrameCreator()
					if not hasattr(self, "panel") or self.panel is None:
						self.panel=self.makePanel()
		if self.isValid:
			serialize_internal_state_of_attr(archive, full, self, "creator")
			serialize_internal_state_by_func(archive, full, "panel", self.panel.serializeInternalState)
	def validate(self):
		super().validate()
		self.fps=60#フレームレート。現在は無効にしている。
		self.isValid=True
		if self.initial_position is not None:
			import os
			old = os.getenv('SDL_VIDEO_WINDOW_POS',None)
			os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int(self.initial_position[0])},{int(self.initial_position[1])}"
			if old is not None:
				os.environ['SDL_VIDEO_WINDOW_POS'] = old
		pygame.init()
		self.window=pygame.display.set_mode((self.width,self.height),pygame.DOUBLEBUF|pygame.OPENGL|pygame.OPENGLBLIT)
		self.clock=pygame.time.Clock()
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		self.panel=self.makePanel()
	def close(self):
		pygame.quit()
	def onEpisodeBegin(self):
		"""エピソードの開始時(reset関数の最後)に呼ばれる。
		"""
		super().onEpisodeBegin()
		self.creator=self.getDataFrameCreator()
		header=self.creator.makeHeader(self.manager)()
		frame=self.creator.makeFrame(self.manager)()
		if hasattr(self.panel, "onEpisodeBegin"):
			self.panel.onEpisodeBegin(self.as_weak(), header, frame)
		self.display(frame)
	def onStepBegin(self):
		"""gym.Envとしてのstepの開始時に呼ばれる。
		"""
		super().onStepBegin()
	def onStepEnd(self):
		"""gym.Envとしてのstepの最後に呼ばれる。
		Managerは得点計算⇛報酬計算⇛その他コールバック⇛画面表示の順で呼ぶ。
		"""
		super().onStepEnd()
	def onInnerStepBegin(self):
		"""インナーループの各ステップの開始時(controlの前)に呼ばれる。
		"""
		super().onInnerStepBegin()
		for e in pygame.event.get():
			if(e.type==QUIT):
				self.manager.manualDone=True
	def onInnerStepEnd(self):
		"""#インナーループの各ステップの最後(perceiveの後)に呼ばれる。
		"""
		# super().onInnerStepEnd()
		if(self.isValid):
			frame=self.creator.makeFrame(self.manager)()
			self.display(frame)
	def onEpisodeEnd(self):
		"""エピソードの終了時(step関数の最後でdone==Trueの場合)に呼ばれる
		"""
		super().onEpisodeEnd()
		if(self.manager.manualDone):
			pygame.quit()
	def display(self,frame):
		"""画面描画を行う。実際の描画処理はPanelに投げる。
		"""
		self.panel.display(frame)
	def makePanel(self):
		"""画面描画を行うPanelオブジェクトを返す。派生クラスでオーバーライドする。
		"""
		raise NotImplementedError

class PygamePanel:
	"""pygameを用いて画面描画を行うPanelオブジェクトの基底クラス。
	"""
	configKeysToBePopped = []
	def __init__(self,config):
		popped = {}
		for key in self.configKeysToBePopped:
			if key in config:
				popped[key]=config.pop(key)
		self.baseConfig = copy.deepcopy(config)
		self.config = self.merger(nljson(self.baseConfig))()
		self.dynamicConfigPaths = []
		for key, value in popped.items():
			config[key] = value

	def serializeInternalState(self, archive, full: bool):
		pass

	def merger(self, src: nljson):
		if src.is_object():
			if "Default" in src:
				ret = nljson({})
				for k, v in src.items():
					d = nljson(src["Default"]) # nljsonを引数としてnljsonを作成するとコピーになる
					if k != "Default":
						d.merge_patch(v)
					ret[k] = self.merger(d)
				return ret
			else:
				return nljson({k: self.merger(v) for k, v in src.items()})
		else:
			return nljson(src) # nljsonを引数としてnljsonを作成するとコピーになる
	def getUIConfig(self, attr: str, json_pointer: str, *, no_exception: bool=False):
		"""ui_config中で、json_pointerで指定したパスに対応するattrで指定した種類の設定値を取得する。
			json_pointerは'/'区切りである。
			なお、本来のjson pointerは先頭も'/'で始める必要があるが、この関数ではどちらでもよい。
		"""
		if json_pointer.startswith('/'):
			json_pointer = json_pointer[1:]
		splitted = json_pointer.split('/')
		now = self.config
		parent = ""
		for key in splitted:
			isDynamic = parent in self.dynamicConfigPaths
			if not key in now:
				if isDynamic:
					key = "Default"
			if key in now:
				now = now[key]
			else:
				if no_exception:
					return None
				else:
					raise KeyError(json_pointer, attr, key)
			if isDynamic:
				parent += "/" + "*"
			else:
				parent += "/" + key
		if attr in now:
			return now[attr]
		else:
			raise KeyError(json_pointer, attr, attr)
	def intColor(self, floatColor):
		"""実数値(0〜1)の色変数を8ビット整数値(0〜255)に変換する。
		"""
		return [int(c*255) for c in floatColor]
	def floatColor(self, intColor):
		"""8ビット整数値(0〜255)の色変数を実数値(0〜1)に変換する。
		"""
		return [c/255 for c in intColor]
