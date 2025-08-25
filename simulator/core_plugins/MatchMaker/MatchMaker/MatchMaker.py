"""!
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
@package MatchMaker.MatchMaker

@details
複数のPolicyの重み保存、読み込みを管理しつつ、
その勝率等に応じたエピソードごとの対戦カードを生成するための基底クラス群であり、以下の2つのクラスからなる。
1. 一つだけ生成し、全体の管理を行うMatchMaker
2. 実際にエピソードを実行するインスタンスごとに生成し、対戦結果を抽出してMatchMakerに渡すresultを生成するMatchMonitor

@par 使用方法

1. MatchMaker のコンストラクタ引数について

    MatchMaker クラスのコンストラクタはdict型の引数configをとる。必須のキーは以下の通り。
    ```python
    config={
        "restore": None or str, #チェックポイントを読み込む場合、そのパスを指定。
        "weight_pool": str, #重みの保存場所。<policy>-<id>.datのようにPolicy名と重み番号を示すファイル名で保存される。
        "policy_config": { #Policyに関する設定。Policy名をキーとしたdictで与える。
            <Policy's name>: Any, #具象クラスごとの様式に従った各Policyの設定
            ...
        },
        "match_config": Any #対戦カードの生成に関する指定。具象クラスで自由に定める
    }
    ```

2. MatchMaker と MatchMonitorの生成

    学習プログラムにおいて、メインプロセスで一つだけ MatchMaker インスタンスを生成する必要がある。
    また、 MatchMonitor はエピソードを生成するWorkerインスタンスごとに生成する必要がある。

3. resultの伝達

    学習プログラムに適した方法にて、各 MatchMonitor.onEpisodeEndが抽出したエピソードの結果をMatchMaker.onEpisodeEndに供給する必要がある。

4. 対戦カードを環境に反映するための機構の準備

    学習プログラムにおいて、生成された対戦カードを用いて環境の生成を行う機構も必要である。

5. 重みの読み書きや初期化に関する処理の実装

    MatchMaker / MatchMonitorは特定の強化学習ライブラリに依存させないために、重みを直接扱わないようにしている。
    そのため、重みの読み書きや初期化を実際に行う処理は学習プログラム側で準備する必要がある。

@par カスタマイズの方法

実際に MatchMaker を使用するうえでは、環境の仕様に応じて以下の関数群を適宜オーバーライドして使用する。

1. MatchMaker.makeNextMatch のオーバーライド

    対戦グループ(○○リーグのようなイメージ)を表す何らかの変数matchTypeを引数にとり
    何らかの対戦カードを返す関数であり、各ユーザーが独自の対戦方式を実装するものである。

    対戦カードはdictで表現され、場に登場するチームごとにどのPolicyのどの重みを使うかをキー"Policy"と"Weight"により指定する。

    また、同名のPolicyで異なる重みを複数同時に使用したい場合等のために、それらを区別するための"Suffix"を指定可能である。

    重み番号は基本的に
    - 負数(-1): 学習中の現物
    - 0: 学習中重みの最新のコピー(随時更新される)
    - 自然数: 過去のある時点で保存されたコピー

    とすることを想定しているが、 MatchMaker を使用する学習プログラムの書き方次第である。

    また、これら以外に必要な情報があれば適宜追加してもよい。まとめると、以下のような書き方になる。
    ```python
    {
        "teams": {
            <Team's name>: {
                "Policy": str, # teamを動かすPolicyの名前
                "Weight": int, # teamを動かすPolicyの重み番号
                "Suffix": str, # teamを動かすPolicyの接尾辞。同名のPolicyで異なる重みを使用する場合等に指定する。省略は不可。
            }
        },
        "common": dict # teamに依存しない条件(Rule等)を記述する。派生クラスで定義する。
    }
    ```

2. MatchMaker.onEpisodeEnd のオーバーライド

    いずれかの対戦が終わったときに呼び出し、その対戦を実施したインスタンスにおける次の対戦カードを生成して返す関数。

    引数は、終了した対戦のmatchInfoと、対応する MatchMonitor.onEpisodeEnd で生成されたresultを与えるものとする。
    また、返り値として、重み保存要否を記したdictを返す必要がある。
    基本的には重み保存を行うPolicyのみを対象として以下のようなdictとして記述することを想定している。
    ```python
    {
        <Policy's name>: {
            "weight_id": int, # 保存先の重み番号
            "reset": bool, # 保存後に重みをリセットするかどうか
        }
    }
    ```

3. MatchMonitor.onEpisodeEnd のオーバーライド

    環境インスタンスごとに対戦結果を抽出して MatchMaker に渡すresultを生成する関数。

    MatchMaker.onEpisodeEnd で重み保存や次回の対戦カードの生成の判定を行うために必要な情報があれば、環境インスタンスから抽出して返す。

4. MatchMaker.checkInitialPopulation のオーバーライド

    学習開始時の重みをweight_poolに追加するかどうかを返す関数。返り値の形式はonEpisodeEndと同じ。
    用途としては、別の学習済モデルを読み込んで開始したときにその初期状態を対戦候補に含めたいような場合等が考えられる。

5. MatchMaker.get_metrics 関数のオーバーライド

    Tensorboardログ等に記録するための値を格納したdictを返す関数として、必要に応じてオーバーライドする。

6. MatchMaker.initialize, load, saveのオーバーライド

    MatchMakerの初期化やチェックポイントの生成、読み込みを行うための各関数を必要に応じてオーバーライドする。
"""
import os
import copy
import cloudpickle
from collections import defaultdict

class MatchMaker:
    def __init__(self,config):
        restore=config.get("restore",None)
        if(restore is not None):
            """チェックポイントからの再開のときは、
                "restore"キーにチェックポイントのパスを指定し、
                それ以外のキーにconfigの上書きを行いたいキーのみ値を指定する。
            """
            self.load(restore,config)
        else:
            """チェックポイントからの再開でないときは、
                "restore"キーを省略するかNoneとする。
            """
            self.initialize(config)
    def initialize(self,config):
        self.resumed=False
        self.config=copy.deepcopy(config)
        self.weight_pool = self.config.get("weight_pool",os.path.join("./log", 'policies', 'weight_pool')) #重み保存先ディレクトリ
        self.policy_config=self.config.get("policy_config",{})
        self.match_config=self.config.get("match_config",{})
        self.trainingWeights={} #各Policyの学習中重み(のコピー)
        self.trainingWeightUpdateCounts=defaultdict(lambda: -1) #各Policyの学習中重みの更新回数
        self.savers={}
        self.loaders={}
        self.extensions={}
        self.initialWeights={}
    def load(self,path,configOverrider={}):
        """チェックポイントを保存する。その際、configOverriderで与えた項目はconfigを上書きする。
        """
        with open(path,"rb") as f:
            obj=cloudpickle.load(f)
            config=copy.deepcopy(obj["config"])
            config.update(configOverrider)
            self.initialize(config)
            self.resumed=True
        print("MatchMaker has been loaded from ",path)
    def save(self,path):
        """チェックポイントを読み込む。
        """
        with open(path,"wb") as f:
            obj={
                "config":self.config,
            }
            cloudpickle.dump(obj,f)
    def get_weight_pool(self):
        """重み保存先のディレクトリを返す。
        """
        return self.weight_pool
    def get_policy_config(self):
        """policy_configを返す。
        """
        return self.policy_config
    def makeNextMatch(self,matchType,worker_index):
        """対戦カードを生成する。
        引数として、対戦グループ(○○リーグのようなイメージ)を表す何らかの変数matchTypeと、実行先となるWorkerのインスタンスIDも与えられるため、
        複数のグループを使い分ける場合は適宜参照しつつ生成する。。
        {
            "teams": {
                team: {
                    "Policy": str, # teamを動かすPolicyの名前
                    "Weight": int, # teamを動かすPolicyの重み番号
                    "Suffix": str, # teamを動かすPolicyの接尾辞。同名のPolicyで異なる重みを使用する場合等に指定する。省略は不可。
                }
            },
            "common": dict # teamに依存しない条件(Rule等)を記述する。派生クラスで定義する。
        }
        なお、重み番号の指定方法は以下の通り。
            負数(-1): 学習中の現物
            0: 学習中重みの最新のコピー(随時更新される)
            自然数: 過去のある時点で保存されたコピー
        """
        return {
            "Team1":{
                "Policy": "Learner",
                "Weight": -1,
                "Suffix": ""
            }
        }
    #
    # 対戦結果の処理
    #
    def onEpisodeEnd(self,match,result):
        """いずれかのgym.Envにおいて対戦が終わったときに呼ばれ、
        そのインスタンスにおける次の対戦カードを生成して返す関数。
        引数は、終了した対戦のmatchInfoと、対応するMatchMonitorのonEpisodeEndで生成されたresultを与える。
        また、返り値として、重み保存要否を記したdictを返す必要がある。
            populate_config = {
                policyName: {
                    "weight_id": int, # 保存先の重み番号
                    "reset": bool, # 保存後に重みをリセットするかどうか
                }
            }
        なお、重み番号の指定方法は以下の通り。
            負数(-1): 学習中の現物
            0: 学習中重みの最新のコピー(随時更新される)
            自然数: 過去のある時点で保存されたコピー
        """
        populate_config={}
        return populate_config
    def get_metrics(self,match,result):
        """SummaryWriter等でログとして記録すべき値をdictで返す。
        """
        return {}
    #
    # 初期重み保存要否の判定
    #
    def checkInitialPopulation(self):
        """開始時の初期重みをpopulateするかどうかを判定し、必要に応じweight_poolに追加する。
            populate_config = {
                policyName: {
                    "weight_id": int, # 保存先の重み番号
                    "reset": bool, # 保存後に重みをリセットするかどうか
                }
            }
        なお、重み番号の指定方法は以下の通り。
            負数(-1): 学習中の現物
            0: 学習中重みの最新のコピー(随時更新される)
            自然数: 過去のある時点で保存されたコピー
        """
        populate_config={}
        return populate_config
    #
    # 学習中重みの集中管理
    #
    def uploadTrainingWeights(self,trainingWeights):
        """指定したポリシー名の学習中重みを書き込む。
        そのポリシーを学習中のTrainer以外からweight_id=0で最新の重みを読み込もうとする場合には、
        この関数で書き込まれた重みを使用できるが、そのためには適切な頻度で外からこの関数を呼ぶ必要がある。
        """
        for policyName,weight in trainingWeights.items():
            self.trainingWeights[policyName]=weight
            self.trainingWeightUpdateCounts[policyName]+=1
    def getTrainingWeightUpdateCount(self,policyName):
        """指定したポリシー名の学習中重みの更新回数を返す。未登録が-1、初期状態が0である。
        """
        return self.trainingWeightUpdateCounts[policyName]
    def getLatestTrainingWeight(self,policyName):
        """指定したポリシー名の学習中重み(のコピー)を返す。
        """
        return self.trainingWeights.get(policyName,None)
    def registerSaverLoader(self, policyName, saver, loader, extension):
        """重みの読み書き方法を登録する。
        Registers saver and loader of weights of policyName.

        Args:
            policyName (str):
            saver (Callable[[Any,str],None]): 1st argument is weight object. 2nd argument is stem string (file name without extension.)
            loader (Callable[[str],[Any]]): The argument string is stem (file name without extension.)
            extension (str): suffix of weight file (extension)
        """
        self.savers[policyName]=saver
        self.loaders[policyName]=loader
        self.extensions[policyName]=extension
    def getInitialWeight(self, policyName):
        """指定したpolicyNameに対する初期重みを返す。
        Returns initial weight for policyName.

        Args:
            policyName (str):
        """
        if not policyName in self.initialWeights:
            initial_weight_path = self.policy_config[policyName].get('initial_weight', None)
            if(initial_weight_path is not None):
                initial_weight = self.loaders[policyName](initial_weight_path)
            else:
                initial_weight = None
            self.initialWeights[policyName] = initial_weight
        return self.initialWeights[policyName]
    def loadWeight(self, policyName, weight_id):
        """指定したpolicyNameとweight_idに対応する重みを読み込んで返す。
        Returns weight corresponding to policyName and weight_id

        Args:
            policyName (str):
            weight_id (int):
        """
        fileName=policyName+"-"+str(weight_id)+self.extensions[policyName]
        srcPath = os.path.join(self.weight_pool, fileName)
        return self.loaders[policyName](srcPath)
    def saveWeight(self, policyName, weight_id, weight):
        """weightをpolicyNameとweight_idに対応する重みとして保存する。
        Saves weight as the weight corresponding to policyName and weight_id

        Args:
            policyName (str):
            weight_id (int):
            weight (Any):
        """
        fileName=policyName+"-"+str(weight_id)+self.extensions[policyName]
        dstPath = os.path.join(self.weight_pool, fileName)
        self.savers[policyName](weight,dstPath)


class MatchMonitor:
    def __init__(self, env):
        self.env=env
    def onEpisodeBegin(self):
        pass
    def onEpisodeEnd(self,matchType):
        """MatchMakerのonEpisodeEndで使用するresultを生成して返す。
        引数として、この対戦を生成した対戦グループ(○○リーグのようなイメージ)を表す何らかの変数matchTypeも与えられる。
        """
        result = {}
        return result
