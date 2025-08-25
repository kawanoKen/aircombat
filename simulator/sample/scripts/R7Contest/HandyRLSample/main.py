# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
# 
# * Discrete以外のaction_spaceに対応
# * MatchMakerによる行動判断モデルの動的選択に対応
# * Imitationに対応
# * Ape-X型のε-greedyに対応
# * SummaryWriterによるTensorboard形式のログ出力に対応
# * turn_basedな環境に非対応
# * evalモードに非対応
# * モデル内の補助タスク(教師あり/教師なし)に対応
# * ReplayBufferを分離し、優先度付き経験再生を使用可能に
# * rayによる並列化に変更
# * 複数のLearnerを連接可能に
# * agent.pyにおいて温度パラメータによる焼きなましを無効化
# * 学習率を固定化可能に
# * eval_rate(学習用と評価用のエピソードの割合)を固定化可能に

import sys
import yaml
import ray
from ray._private.ray_constants import NODE_DEFAULT_IP
from ASRCAISim1.plugins.HandyRLUtility.model import DummyInternalModel
from R7ContestSample.R7ContestTorchNNSampleForHandyRL import R7ContestTorchNNSampleForHandyRL
from R7ContestModels.R7ContestTwoTeamCombatMatchMaker import R7ContestTwoTeamCombatMatchMaker
from ASRCAISim1.plugins.MatchMaker.TwoTeamCombatMatchMaker import TwoTeamCombatMatchMonitor
from ASRCAISim1.plugins.HandyRLUtility.distribution import getActionDistributionClass

custom_classes={
    # models
    "R7ContestTorchNNSampleForHandyRL": R7ContestTorchNNSampleForHandyRL,
    "DummyInternalModel": DummyInternalModel,
    # match maker
    "R7ContestTwoTeamCombatMatchMaker": R7ContestTwoTeamCombatMatchMaker,
    "TwoTeamCombatMatchMonitor": TwoTeamCombatMatchMonitor,
    # action distribution class getter
    "actionDistributionClassGetter": getActionDistributionClass,
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please set config yaml path.')
        exit(1)
    with open(sys.argv[1]) as f:
        args = yaml.safe_load(f)

    #rayの初期化
    namespace=args.get("ray_args",{}).get("namespace","ASRCAISim1")
    head_ip_address=args.get("ray_args",{}).get("head_ip_address","auto")
    entrypoint_ip_address=args.get("ray_args",{}).get("entrypoint_ip_address",NODE_DEFAULT_IP)
    try:
        ray.init(address=head_ip_address,namespace=namespace,_node_ip_address=entrypoint_ip_address)
    except:
        print("Warning: Failed to init with the given head_ip_address. A new cluster will be launched instead.")
        ray.init(namespace=namespace,_node_ip_address=entrypoint_ip_address)
    node_ip_address=ray._private.services.get_node_ip_address()
    if not "ray_args" in args:
        args["ray_args"]={}
    args["ray_args"].update({
        "namespace": namespace,
        "head_ip_address": head_ip_address,
        "entrypoint_ip_address": entrypoint_ip_address,
        "node_ip_address": node_ip_address,
    })

    args["custom_classes"] = custom_classes
    args["teams"] = args['match_maker_args'].get('teams',['Blue','Red'])

    #環境の設定(yamlで指定できないもの)
    import os
    cwd=os.getcwd()
    args["env_args"]["env_config"]["cwd"]=cwd
    def importer():
        #バックグラウンドで走るSimulationManager用の追加初期化関数。
        pass
    args["env_args"]["env_config"]["additionalInitializer"]=importer

    from handyrl.train import train_main as main
    main(args)
