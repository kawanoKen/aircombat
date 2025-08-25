# Copyright (c) 2021-2023 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#----------------------------------------------------------------
# ASRCAISim1プラグイン用の__init__.pyのテンプレート
# __init__.py template for ASRCAISim1's plugin
#----------------------------------------------------------------

##############################################
# ASRCAISim1プラグインの共通初期化処理
#     (1) 依存するプラグインをインポート(し、通常のインポートと同様に自身のattributeに追加)
#     (2) C++モジュールがあればインポートしてそのattributeを自身のattributeに追加
#     (3) FactoryHelperインスタンスを'factoryHelper'としてattributeに追加
# Common initialization for ASRCAISim1 plugins
#     (1) Import plugins required by this plugin (and set as the plugin's attributes as usual.)
#     (2) If a C++ module exists, import it and set its attributes as the plugin's attributes.
#     (3) Add a FactoryHelper instance as 'factoryHelper' attribute.
##############################################
import sys
import ASRCAISim1
ASRCAISim1.common_init_for_plugin(sys.modules[__name__], globals())

##############################################
# ここから各ASRCAISim1プラグイン独自の初期化を記述する
# Individual initialization for each ASRCAISim1 plugin starts from here
##############################################

#コンフィグの読み込み
from pathlib import Path
import glob
moduleDir=Path(__file__).resolve().parent
configFiles=glob.glob(str(moduleDir.joinpath("config","factory/**/*.json")),recursive=True)
for c in configFiles:
    factoryHelper.addDefaultModelsFromJsonFile(c)
