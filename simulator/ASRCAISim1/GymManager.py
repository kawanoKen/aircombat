"""!
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
@package ASRCAISim1.GymManager
"""
from typing import (
	Any,
	Callable,
	Dict,
	Optional,
	Tuple,
)
import gymnasium as gym
from ASRCAISim1.core import (
	SimulationManager,
	ExpertWrapper
)
from ASRCAISim1.policy import StandalonePolicy

class GymManager(gym.Env):
	"""!
		@brief gym.Envを継承した、SimulationManagerのラッパークラス。
	"""

	def __init__(self,context: Dict[str,Any]):
		"""! @brief コンストラクタ。

			@details Ray.RLLibのインターフェースに準じて、コンストラクタに与える引数はrayのEnvContextに相当するdict型のcontextのみ。

			context={
				"config":Union[Dict[str,Any]|List[Union[Dict[str,Any]|str]]],
				"worker_index":int,
				"vector_index":int,
				"overrider":Optional[Callable[[Dict[str,Any],int,int],Dict[str,Any]]],
				"manager_class":Optional[Type],
			}

			Args:
				context (dict): context["config"]にSimulationManagerに渡すjsonを持たせる。
								また、必要に応じて、context["overrider"]にworker_index及びvector_indexに応じたconfig置換関数(std::function<nl::json(const nl::json&,int,int)>相当)を与える。
								SimulationManager以外のクラスを使用したい場合は、context["manager_class"]としてクラスオブジェクトを指定する。
		"""
		self.context=context
		self.worker_index=context.get("worker_index",0)
		self.vector_index=context.get("vector_index",0)
		manager_class=context.get("manager_class",None)
		if manager_class is None:
			manager_class = SimulationManager
		self.manager=manager_class(context["config"],self.worker_index,self.vector_index,context.get("overrider",None))
	def render(self,mode="Default"):
		"""!
			@brief 環境の描画を行う。
			@note gym.Envのインターフェースに合わせて実装しているが、
			現時点では、Viewerクラスが独自に描画するため、render関数としては動作しない。
		"""
		pass
	def reset(
		self,
		*,
		seed: Optional[int] = None,
		options: Optional[dict] = None,
	) -> Tuple[Dict[str, Any], Dict[str, dict]]:
		"""! @brief 環境のリセットを行う。
		"""
		ret=self.manager.reset(seed=seed,options=options)
		self.get_observation_space()
		self.get_action_space()
		return ret
	def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
		"""! @brief 環境を1ステップすすめる。
		"""
		return self.manager.step(action)
	@property
	def observation_space(self):
		return self.get_observation_space()
	def get_observation_space(self):
		"""! @brief gym.Envとして外部に見せる状態空間を計算する。

		@details gymのインターフェースを厳密に守ると、今のところ状態空間と行動空間は固定であり、かつメソッドでなくプロパティとして保持しなければならず、stable baselinesもその前提で実装されている。
		しかし、このManagerはエピソードごとに可変とすることもできてしまうため、
		LearnerとExpertの種類・数が固定となるようにconfigの作成時に気をつけるものとする。
		"""
		self._observation_space=gym.spaces.Dict(self.manager.get_observation_space())
		return self._observation_space
	@property
	def action_space(self):
		return self.get_action_space()
	def get_action_space(self):
		"""g! @brief ym.Envとして外部に見せる行動空間を計算する。

		@details gymのインターフェースを厳密に守ると、今のところ状態空間と行動空間は固定であり、かつメソッドでなくプロパティとして保持しなければならず、stable baselinesもその前提で実装されている。
		しかし、このManagerはエピソードごとに可変とすることもできてしまうため、
		LearnerとExpertの種類・数が固定となるようにconfigの作成時に気をつけるものとする。
		"""
		self._action_space=gym.spaces.Dict(self.manager.get_action_space())
		return self._action_space
	def getManagerConfig(self):
		return self.manager.getManagerConfig()
	def getFactoryModelConfig(self, withDefaultModels=True):
		return self.manager.getFactoryModelConfig(withDefaultModels)
	def requestReconfigure(self,config1: Dict[str,Any], config2: Optional[Dict[str,Any]]=None):
		"""! @brief configの変更を予約する。現在実施中のエピソードが終了後、次のreset時に反映される。
		
		@details 反映は、jsonのmerge patch(dictのupdateを再帰的に行うことに類似した処理)により行われるため、対応するキーの値が置き換わるのみであり、dictそのものの置き換えではない。

		Args:
			config1 (dict | None): 1引数で呼び出す場合はコンストラクタ引数のcontext["config"]に相当する形式で与え、
			                       2引数で呼び出す場合はcontext["config"]["Manager"]に相当する形式で与える。
			config2 (dict | None): 2引数で呼び出す場合はコンストラクタ引数のcontext["config"]["Factory"]に相当する形式で与える。
		"""
		if config2 is None:
			self.manager.requestReconfigure(config1)
		else:
			self.manager.requestReconfigure(config1, config2)
	def setViewerType(self,viewerType):
		"""! @brief Viewerのモデル名を指定する。
		"""
		self.manager.setViewerType(viewerType)
	def configure(self,config1: Optional[Dict[str,Any]]=None, config2: Optional[Dict[str,Any]]=None):
		"""! @brief 与えられたconfigを用いてインスタンスの初期化を行う。
		
		@details 引数なしで呼んだ場合は現在のconfigを用いてインスタンスの初期化を行う。

		Args:
			config1 (dict | None): 1引数で呼び出す場合はコンストラクタ引数のcontext["config"]に相当する形式で与え、
			                       2引数で呼び出す場合はcontext["config"]["Manager"]に相当する形式で与える。
			config2 (dict | None): 2引数で呼び出す場合はコンストラクタ引数のcontext["config"]["Factory"]に相当する形式で与える。
		"""
		if config2 is None:
			if config1 is None:
				self.manager.configure()
			else:
				self.manager.configure(config1)
		else:
			self.manager.configure(config1, config2)
	def reconfigureIfRequested(self):
		"""! @brief requestReconfigureで予約されたconfig変更があればそれを反映してインスタンスの初期化を行う。

		@details 必要に応じてreset内で呼ばれるため手動で呼ぶ必要はない。
		"""
		self.manager.reconfigureIfRequested()
	def reconfigure(self,fullConfigPatch):
		"""! @brief 与えられたconfigPatchによるconfig変更を反映してインスタンスの初期化を行う。

		@details このとき、requestReconfigureで予約されたconfig変更があればそれも追加で反映してインスタンスの初期化を行う。
		Args:
			fullConfigPatch (dict): コンストラクタ引数のcontext["config"]に相当する形式で与える。
		"""
		self.manager.reconfigure(fullConfigPatch)
	def reconfigure(self,config1: Dict[str,Any], config2: Optional[Dict[str,Any]]=None):
		"""! @brief 与えられたconfigPatchによるconfig変更を反映してインスタンスの初期化を行う。

		@details このとき、requestReconfigureで予約されたconfig変更があればそれも追加で反映してインスタンスの初期化を行う。

		Args:
			config1 (dict | None): 1引数で呼び出す場合はコンストラクタ引数のcontext["config"]に相当する形式で与え、
			                           2引数で呼び出す場合はコンストラクタ引数のcontext["config"]["Manager"]に相当する形式で与える。
			config2 (dict | None): 2引数で呼び出す場合はコンストラクタ引数のcontext["config"]["Factory"]に相当する形式で与える。
		"""
		if config2 is None:
			self.manager.reconfigure(config1)
		else:
			self.manager.reconfigure(config1, config2)

def getDefaultPolicyMapper():
	"""! @brief エージェントとポリシーのマッピング関数を生成する。

		@details 基本的には環境側のconfigでポリシー名を指定しておくものとする。
		環境の出力となる辞書のキーとなるAgentのfullNameは、fullName=agentName:modelName:policyNameの形式としている。
	"""
	def ret(fullName: str):
		agentName,modelName,policyName=fullName.split(":")
		return policyName
	return ret

class SinglizedEnv(GymManager):
	"""! @brief 一つを除いたAgentのactionを内部で処理することによって、シングルエージェント環境として扱うためのラッパークラスの実装例。
	"""

	def __init__(self,context: Dict[str,Any]):
		"""! @brief コンストラクタ
		
			@details
				contextの追加要素={
					"target": Union[Callable[[str],bool],str],
					"policies": Dict[str,StandalonePolicy],
					"policyMapper": Optional[Callable[[str],str]],
					"runUntilAllDone": bool (True if omitted)
				}

				Args:
					target (Union[Callable[[str],bool],str]): observation,actionの入出力対象のAgentを特定するための関数。AgentのfullNameを引数にとってboolを返す関数を与える。最初にTrueとなったAgentを対象とみなす。
						Callableの代わりに文字列を指定した場合、その文字列で始まるかどうか(startswith)を判定条件にすることが可能。
					policies (Dict[str,StandalonePolicy]): 内部でactionを計算するためのStandalonePolicyオブジェクトのdict。キーはpolicyの名称となる。
					policyMapper (Callable[[str],str]): AgentのfullNameから使用するpolicyの名称を計算する。省略した場合はfullNameの末尾にあるpolicyName部分が使われる。
					runUntilAllDone (bool): 対象AgentのdoneがTrueとなっても、done["__all__"]がTrueとなるまで内部でエピソードの計算を継続するか否か。デフォルトはTrue。
		"""
		target_=context["target"]
		if(isinstance(target_,str)):
			self.target=lambda fullName:fullName.startswith(target_)
		elif(isinstance(target_,Callable)):
			self.target=target_
		else:
			raise ValueError("SinglizedEnv requires Callable[[str],bool] or str for 'target' in context.")
		self.targetAgent=None
		self.policies=context.get("policies",{})
		self.originalPolicyMapper=context.get("policyMapper",getDefaultPolicyMapper())
		self.policyMapper=self.originalPolicyMapper
		self.runUntilAllDone=context.get("runUntilAllDone",True)
		super().__init__(context)
	def reset(
		self,
		*,
		seed: Optional[int] = None,
		options: Optional[dict] = None,
	) -> Tuple[Any, dict]:
		self.targetAgent=None
		self.obs, infos=super().reset(seed=seed,options=options)
		self.rewards={}
		self.rewards={k:0.0 for k in self.obs.keys()}
		self.terminateds={k:False for k in self.obs.keys()}
		self.truncateds={k:False for k in self.obs.keys()}
		self.infos={k:None for k in self.obs.keys()}
		self.terminateds["__all__"]=False
		self.truncateds["__all__"]=False
		for p in self.policies.values():
			p.reset()
		if(self.isExpertWrapper):
			self.targetObs=self.targetAgent.imitatorObs
		else:
			self.targetObs=self.obs[self.targetIdentifier]
		return self.targetObs, infos[self.targetIdentifier]
	def step(self,action) -> Tuple[Any, float, bool, bool, dict]:
		observation_space=self.get_observation_space()
		action_space=self.get_action_space()
		actions={k:self.policies[self.policyMapper(k)].step(
			o,
			self.rewards[k],
			self.terminateds[k] or self.truncateds[k],
			self.infos[k],
			k,
			observation_space[k],
			action_space[k]
		) for k,o in self.obs.items() if self.policyMapper(k) in self.policies}
		actions[self.targetIdentifier]=action
		self.obs,self.rewards,self.terminateds,self.truncateds,self.infos=super().step(actions)
		if(self.isExpertWrapper):
			self.targetObs=self.targetAgent.imitatorObs
		else:
			self.targetObs=self.obs[self.targetIdentifier]
		ret=self.targetObs,self.rewards[self.targetIdentifier],self.terminateds[self.targetIdentifier],self.truncateds[self.targetIdentifier],self.infos[self.targetIdentifier]
		dones={k:self.terminateds[k] or self.truncateds[k] for k in self.terminateds}
		if(dones[self.targetIdentifier] and not dones["__all__"]):
			if(self.runUntilAllDone):
				while(not dones["__all__"]):
					observation_space=self.get_observation_space()
					action_space=self.get_action_space()
					actions={k:self.policies[self.policyMapper(k)].step(o,k,observation_space[k],action_space[k]) for k,o in self.obs.items() if self.policyMapper(k) in self.policies}
					if(self.targetIdentifier in actions):
						actions.pop(self.targetIdentifier)
					self.obs,self.rewards,self.terminateds,self.truncateds,self.infos=super().step(actions)
					dones={k:self.terminateds[k] or self.truncateds[k] for k in self.terminateds}
			else:
				self.manager.stopEpisodeExternally()
		return ret
	def get_observation_space(self):
		self.manager.get_observation_space()
		if(self.targetAgent is None):
			self.setupTarget()
		self._observation_space=self.targetAgent.observation_space()
		return self._observation_space
	def get_action_space(self):
		self.manager.get_action_space()
		if(self.targetAgent is None):
			self.setupTarget()
		self._action_space=self.targetAgent.action_space()
		return self._action_space
	def setupTarget(self):
		agents={agent().getFullName():agent() for agent in self.manager.getAgents()}
		for fullName,agent in agents.items():
			if(self.target(fullName)):
				self.targetIdentifier=fullName
				self.targetAgent=agent
				self.isExpertWrapper=isinstance(self.targetAgent,ExpertWrapper)
				break
		assert self.targetAgent is not None

class SimpleEvaluator:
	"""! @brief 全Agentの行動判断をStandalonePolicyにより内部で処理する評価用環境の最小構成。

		@details 勝敗や得点の抽出等、結果の取得・利用はCallbackを用いてもよいし、このクラスを改変して実装してもよい。
		Args:
			context (Dict[str,Any]): GymManagerに与えるcontext。
			policies (Dict[str,StandalonePolicy]): 内部でactionを計算するためのStandalonePolicyオブジェクトのdict。キーはpolicyの名称となる。
			policyMapper (Callable[[str],str]): AgentのfullNameから使用するpolicyの名称を計算する。省略した場合はfullNameの末尾にあるpolicyName部分が使われる。
	"""
	def __init__(self,context: Dict[str,Any], policies: Dict[str,StandalonePolicy], policyMapper: Callable[[str],str]=getDefaultPolicyMapper()):
		self.env=GymManager(context)
		self.policies=policies
		self.policyMapper=policyMapper
	def run(
		self,
		*,
		seed: Optional[int] = None,
		options: Optional[dict] = None,
	):
		"""! @brief エピソードを1回実行する。
		"""
		obs, info = self.env.reset(seed=seed,options=options)
		rewards={k:0.0 for k in obs.keys()}
		terminateds = {k: False for k in obs.keys()}
		truncateds = {k: False for k in obs.keys()}
		infos={k:None for k in obs.keys()}
		for p in self.policies.values():
			p.reset()
		terminateds["__all__"]=False
		truncateds["__all__"]=False
		while not (terminateds["__all__"] or truncateds["__all__"]):
			observation_space=self.env.get_observation_space()
			action_space=self.env.get_action_space()
			actions={k:self.policies[self.policyMapper(k)].step(
				o,
				rewards[k],
				terminateds[k] or truncateds[k],
				infos[k],
				k,
				observation_space[k],
				action_space[k]
			) for k,o in obs.items() if self.policyMapper(k) in self.policies}
			obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
