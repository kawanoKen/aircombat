# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import json
from ASRCAISim1.core import factoryHelper

#Loggerの登録

from ASRCAISim1.logger.LoggerSample import BasicLogger
factoryHelper.addPythonClass('Callback','BasicLogger',BasicLogger)

from ASRCAISim1.logger.MultiEpisodeLogger import MultiEpisodeLogger
factoryHelper.addPythonClass('Callback','MultiEpisodeLogger',MultiEpisodeLogger)

try:
	from ASRCAISim1.logger.GodViewLogger import GodViewLogger
	factoryHelper.addPythonClass('Callback','GodViewLogger',GodViewLogger)
except:
	pass

from ASRCAISim1.logger.GUIStateLogger import GUIStateLogger
factoryHelper.addPythonClass('Callback','GUIStateLogger',GUIStateLogger)
