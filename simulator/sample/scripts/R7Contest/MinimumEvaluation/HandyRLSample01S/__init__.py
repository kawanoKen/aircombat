# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#HandyRLを用いた学習サンプルで学習したモデルの登録方法例
import os,json
import yaml
import ASRCAISim1
from ASRCAISim1.plugins.HandyRLUtility.StandaloneHandyRLPolicy import StandaloneHandyRLPolicy
from ASRCAISim1.plugins.HandyRLUtility.distribution import getActionDistributionClass
#①Agentクラスオブジェクトを返す関数を定義
"""
以下はサンプルのAgentクラスを借りてくる場合の例
"""
def getUserAgentClass(args={}):
    from R7ContestSample import R7ContestAgentSample01
    return R7ContestAgentSample01

#②Agentモデル登録用にmodelConfigを表すjsonを返す関数を定義
"""
なお、modelConfigとは、Agentクラスのコンストラクタに与えられる二つのjson(dict)のうちの一つであり、設定ファイルにおいて
{
    "Factory":{
        "Agent":{
            "modelName":{
                "class":"className",
                "config":{...}
            }
        }
    }
}の"config"の部分に記載される{...}のdictが該当する。
"""    
def getUserAgentModelConfig(args={}):
    return json.load(open(os.path.join(os.path.dirname(__file__),"agent_config.json"),"r"))

#③Agentの種類(一つのAgentインスタンスで1機を操作するのか、陣営全体を操作するのか)を返す関数を定義
"""AgentがAssetとの紐付けに使用するportの名称は本来任意であるが、
　簡単のために1機を操作する場合は"0"、陣営全体を操作する場合は"0"〜"機数-1"で固定とする。
"""
def isUserAgentSingleAsset(args={}):
    #1機だけならばTrue,陣営全体ならばFalseを返すこと。
    return True

#④StandalonePolicyを返す関数を定義
def getUserPolicy(args={}):
    from R7ContestSample.R7ContestTorchNNSampleForHandyRL import R7ContestTorchNNSampleForHandyRL
    import glob
    model_config=yaml.safe_load(open(os.path.join(os.path.dirname(__file__),"model_config.yaml"),"r"))
    weightPath=None
    if args is not None:
        weightPath=args.get("weightPath",None)
    if weightPath is not None:
        weightPath=os.path.join(os.path.dirname(__file__),weightPath)
        if not os.path.exists(weightPath):
            weightPath=None
    if weightPath is None:
        cwdWeights=glob.glob(os.path.join(os.path.dirname(__file__),"*.pth"))
        weightPath=cwdWeights[0] if len(cwdWeights)>0 else None
    if weightPath is None:
        print("Warning: Model weight file was not found. ",__name__)
    isDeterministic=False #決定論的に行動させたい場合はTrue、確率論的に行動させたい場合はFalseとする。
    return StandaloneHandyRLPolicy(R7ContestTorchNNSampleForHandyRL,model_config,weightPath,getActionDistributionClass,isDeterministic)

#⑤Blue側だった場合の初期条件を返す関数を定義
def getBlueInitialState(args={}):
    #(1)任意の値を指定
    state=[
        {"pos":[-20000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
        {"pos":[-10000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
        {"pos":[10000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
        {"pos":[20000.0,100000.0,-10000.0], "vel":270.0, "heading":270.0,},
    ]
    '''
    #(2)一定領域内ランダムで指定
    state=[{
        "pos":gym.spaces.Box(low=np.array([-20000.0,95000.0,-12000.0]),high=np.array([20000.0,105000.0,-6000.0]),dtype=np.float64).sample().tolist(),
        "vel":260.0+20.0*np.random.rand(),
        "heading": 225.0+90.0*np.random.rand(),
    } for i in range(4)]
    '''
    return state
