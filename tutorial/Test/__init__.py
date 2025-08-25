import os,json
import yaml
from ASRCAISim1.policy import StandalonePolicy
from ASRCAISim1.plugins.HandyRLUtility.StandaloneHandyRLPolicy import StandaloneHandyRLPolicy # type: ignore
from ASRCAISim1.plugins.HandyRLUtility.distribution import getActionDistributionClass # type: ignore

def getUserAgentClass(args={}):
    from .MyAgent import SampleAgent
    return SampleAgent


def getUserAgentModelConfig(args={}):
    configs=json.load(open(os.path.join(os.path.dirname(__file__),"agent_config.json"),"r"))

    return configs


def isUserAgentSingleAsset(args={}):
    return False

#デバッグ用にテスト出力したいとき
# class DummyPolicy(StandalonePolicy):
#     def step(self,observation,reward,done,info,agentFullName,observation_space,action_space):
#         actions = action_space.sample()
#         b = []
#         for a in actions:
#             d = {k: int(v) for k, v in a.items()}
#             b.append(d)
#         print(b)
#         return b


# def getUserPolicy(args={}):
#     return DummyPolicy()


def getUserPolicy(args={}):
    from R7ContestSample.R7ContestTorchNNSampleForHandyRL import R7ContestTorchNNSampleForHandyRL
    import glob
    model_config=yaml.safe_load(open(os.path.join(os.path.dirname(__file__),"model_config.yaml"),"r"))
    weightPath=None
    if args is not None:
        weightPath=args.get("weightPath",None)
    if weightPath is not None:
        weightPath=os.path.join(os.path.dirname(__file__),weightPath)
        print(weightPath)
        if not os.path.exists(weightPath):
            weightPath=None
    if weightPath is None:
        cwdWeights=glob.glob(os.path.join(os.path.dirname(__file__),"*.pth"))
        weightPath=cwdWeights[0] if len(cwdWeights)>0 else None
    if weightPath is None:
        print("Warning: Model weight file was not found. ",__name__)
    isDeterministic=False #決定論的に行動させたい場合はTrue、確率論的に行動させたい場合はFalseとする。
    return StandaloneHandyRLPolicy(R7ContestTorchNNSampleForHandyRL,model_config,weightPath,getActionDistributionClass,isDeterministic)
