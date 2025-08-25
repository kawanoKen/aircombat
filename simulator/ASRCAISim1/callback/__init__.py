# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import json
from ASRCAISim1.core import factoryHelper


#Callbackの登録
from ASRCAISim1.callback.EpisodeMonitor import EpisodeMonitor
factoryHelper.addPythonClass('Callback','EpisodeMonitor',EpisodeMonitor)

