# Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from gymnasium import spaces
from ASRCAISim1.core import (
	SingleAssetAgent,
)

class your_class(SingleAssetAgent):
	def initialize(self):
		super().initialize()

	def serializeInternalState(self, archive, full: bool):
		super().serializeInternalState(archive, full)

	def validate(self):
		super().validate()

	def observation_space(self):
		return spaces.Discrete(1)

	def makeObs(self):
		return 1

	def action_space(self):
		return spaces.Discrete(1)

	def deploy(self,action):
		pass

	def control(self):
		pass

	def convertActionFromAnother(self,decision,command):#模倣対象の行動または制御出力と近い行動を計算する
		return 1

	def controlWithAnotherAgent(self,decision,command):
		#基本的にはオーバーライド不要だが、模倣時にActionと異なる行動を取らせたい場合に使用する。
		self.control()
		#例えば、以下のようにcommandを置換すると射撃のみexpertと同タイミングで行うように変更可能。
		#self.commands[self.parent.getFullName()]["weapon"]=command[self.parent.getFullName()]["weapon"]
