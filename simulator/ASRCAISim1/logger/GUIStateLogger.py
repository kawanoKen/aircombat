# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import bz2
import datetime
import numpy as np
import os
import pickle
from ASRCAISim1.core import (
	Callback,
	getValueFromJsonKRD,
	GUIDataFrameCreator,
	serialize_attr_with_type_info,
	serialize_internal_state_of_attr,
	isInputArchive,
)

class GUIStateLogger(Callback):
	"""GUI表示に必要なシミュレーション情報をログとして保存するためのLogger。
	保存されたログをPygameViewerLoaderの派生クラスで読み込むことで表示が可能。
	"""
	def initialize(self):
		super().initialize()
		self.prefix=getValueFromJsonKRD(self.modelConfig,"prefix",self.randomGen,"")
		self.episodeInterval=getValueFromJsonKRD(self.modelConfig,"episodeInterval",self.randomGen,1)
		self.innerInterval=getValueFromJsonKRD(self.modelConfig,"innerInterval",self.randomGen,1)
		self.episodeCounter=0
		self.innerCounter=0
		self.timeStamp=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
		self.hasEpisodeBegun=False
		self.isFileCreated=False
		self.fileAbsPath=""
		self.fo=None
		self.pickler=None

	def serializeInternalState(self, archive, full: bool):
		super().serializeInternalState(archive, full)
		if full:
			serialize_attr_with_type_info(archive, self
				,"prefix"
				,"episodeInterval"
				,"innerInterval"
				,"timeStamp"
			)
		serialize_attr_with_type_info(archive, self
			,"episodeCounter"
			,"innerCounter"
			,"isFileCreated"
			,"hasEpisodeBegun"
		)
		if self.isFileCreated:
			serialize_attr_with_type_info(archive, self, "fileAbsPath")
			if isInputArchive(archive):
				self.fo=bz2.BZ2File(self.fileAbsPath,"ab",compresslevel=9)
				self.pickler=pickle.Pickler(self.fo,protocol=4)
		else:
			if isInputArchive(archive):
				self.fileAbsPath=""
				self.fo=None
				self.pickler=None
		if self.hasEpisodeBegun:
			if isInputArchive(archive):
				self.creator=self.getDataFrameCreator()
			serialize_internal_state_of_attr(archive, full, self, "creator")
			serialize_attr_with_type_info(archive, self, "header")
			serialize_attr_with_type_info(archive, self, "initialFrame")
	def onEpisodeBegin(self):
		"""エピソードの開始時(reset関数の最後)に呼ばれる。
		"""
		self.hasEpisodeBegun=True
		self.innerCounter=0
		self.episodeCounter+=1
		self.creator=self.getDataFrameCreator()
		self.header=self.creator.makeHeader(self.manager)
		self.initialFrame=self.creator.makeFrame(self.manager)

	def onInnerStepEnd(self):
		"""#インナーループの各ステップの最後(perceiveの後)に呼ばれる。
		"""
		self.innerCounter+=1
		if(self.episodeCounter%self.episodeInterval==0):
			if(self.innerCounter%self.innerInterval==0):
				if(not self.isFileCreated):
					self.fileAbsPath=os.path.abspath(
						self.prefix+"_"+self.timeStamp+"_e{:04d}.dat".format(self.episodeCounter)
					)
					os.makedirs(os.path.dirname(self.fileAbsPath),exist_ok=True)
					self.fo=bz2.BZ2File(self.fileAbsPath,"wb",compresslevel=9)
					self.pickler=pickle.Pickler(self.fo,protocol=4)
					self.isFileCreated=True
					self.pickler.dump(self.header)
					self.pickler.dump(self.initialFrame)
					self.pickler.clear_memo()
					self.header=None
					self.initialFrame=None
				self.pickler.dump(self.creator.makeFrame(self.manager))
				self.pickler.clear_memo()

	def onEpisodeEnd(self):
		"""エピソードの終了時(step関数の最後でdone==Trueの場合)に呼ばれる
		"""
		self.fo.close()
		self.fo=None
		self.pickler=None
		self.isFileCreated=False
		self.creator=None
		self.hasEpisodeBegun=False

	def getDataFrameCreator(self):
		"""データフレームの生成用オブジェクトを返す。
		"""
		return GUIDataFrameCreator()
