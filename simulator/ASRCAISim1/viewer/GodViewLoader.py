# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

import os
from ASRCAISim1.viewer.PygameViewerLoader import PygameViewerLoader
from ASRCAISim1.viewer.GodView import GodViewPanel

class GodViewLoader(PygameViewerLoader):
	"""GodViewと同等の表示を、GUIStateLoggerにより保存されたログを読み込んで行いつつ、
	連番画像又は動画として保存するためのクラス。(保存の有無は選択可能。)
	インターフェースはCallbackに準じているが、SimulationManagerとは独立に動くものである。
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
	defaultPanelConfigPath = os.path.join(os.path.dirname(__file__),"../config","GodViewUIConfig.json")
	panelClass = GodViewPanel
	def __init__(self,modelConfig={},instanceConfig={}):
		super().__init__(modelConfig,instanceConfig)
		# 描画用のCRSの指定に関する情報を取得
		self.crsConfig=None
		if "crs" in self.instanceConfig:
			self.crsConfig=self.instanceConfig["crs"]
		elif "crs" in self.modelConfig:
			self.crsConfig=self.modelConfig["crs"]
		self.panelConfig.update({
			"width": self.width,
			"height": self.height,
			"crsConfig": self.crsConfig,
		})
	def makePanel(self):
		"""画面描画を行うPanelオブジェクトを返す。
		"""
		return self.panelClass(self.panelConfig)
