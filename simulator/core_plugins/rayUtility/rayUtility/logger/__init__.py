#-*-coding:utf-8-*-
import json
from ASRCAISim1.core import factoryHelper

#Loggerの登録
from .ExpertTrajectoryWriter import ExpertTrajectoryWriter
factoryHelper.addPythonClass('Callback','ExpertTrajectoryWriter',ExpertTrajectoryWriter)

from .RayMultiEpisodeLogger import RayMultiEpisodeLogger
factoryHelper.addPythonClass('Callback','RayMultiEpisodeLogger',RayMultiEpisodeLogger)
