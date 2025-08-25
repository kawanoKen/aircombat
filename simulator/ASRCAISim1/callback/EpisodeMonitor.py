# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
import sys,os,time
import numpy as np
import datetime
from ASRCAISim1.core import (
    Callback,
    serialize_attr_with_type_info,
)

class EpisodeMonitor(Callback):
	"""エピソード終了時に得点と実行時間を出力するだけのデバッグ向けクラス。
	"""
	def initialize(self):
		super().initialize()
		self.episodeCounter=0
	def serializeInternalState(self, archive, full: bool):
		super().serializeInternalState(archive, full)
		serialize_attr_with_type_info(archive, self,
			"episodeCounter",
			"startT"
		)
	def onEpisodeBegin(self):
		self.episodeCounter+=1
		self.startT=time.time()
	def onStepEnd(self):
		#if(self.manager.tick%100==0):
		#	print("t=",self.manager.t)
		pass
	def onEpisodeEnd(self):
		endT=time.time()
		print("Episode",self.episodeCounter," finished at t=",self.manager.getElapsedTime(),",","with  score=", {k: v for k, v in self.manager.scores.items()}," in ",endT-self.startT," seconds.")
