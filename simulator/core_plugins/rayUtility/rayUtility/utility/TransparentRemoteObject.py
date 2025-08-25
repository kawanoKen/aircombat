# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#
# バックグラウンドで処理させたいオブジェクトをray.Actorとして配置しつつ元のオブジェクトと同じインターフェースで透過的に操作するためのユーティリティ
# ====== basic usage ======
# (1) クラスclsのオブジェクトをActorとして配置する際はクラスメソッドのcreateを用いる。
#     その際、Actor生成時のoptionをdictで与える。例えば、max_concurrencyやnum_cpusなどが指定可能である。
#     なお、max_concurrencyの値1より大きくするとThreadedActorとなり複数のプロセスから同時に呼び出すことができるようになるが、
#     torchのno_gradのようにスレッドローカルな処理の場合は正しく動作しないため、そのような処理を用いる場合は1とする必要がある。
#     tro = TransparentRemoteObject.create(cls,{"max_concurrency":16,"num_cpus":1.0},*args,**kwargs)
# (2) 「非同期でない形での」属性への操作はweakproxyと同様に、本来のclsオブジェクトと同じ書式で可能。例えば、以下のように記述できる。
#     print(tro.hoge)
#     tro.hoge = 1
#     tro.hoge += 2
#     ret = tro.fuga(tro.hoge, x = 3)
# (3) メンバメソッドを非同期実行したい場合(バックグラウンド化する本来の目的)は、rayのActorと同じ形でremoteメソッドを用いる。
#     ref = back.async_func.remote(some_args)
# (4) メンバ変数やメソッドの返り値は、immutableでないものはTransparentRemoteObjectとして得られる。
#     immutableな返り値はその値が直接得られる。
# (5) 中身のオブジェクトの取得はunwrapメソッドを用いる。immutableかそうでないか不明な場合は、unwrap_if_needed(tro)を用いる。
#     actual = tro.unwrap()
#     actual = unwrap_if_needed(tro)
# (6) 非同期実行したメソッドの実際の返り値の取得はray.getの代わりに、返り値に__call__()を呼び出すことで行う。
#     actual = ref()
# (7) 他のプロセスとTransparentRemoteObjectを共有したい場合は、TransparentRemoteObjectInfoを介して行う。
#     TransparentRemoteObjectInfoを受け取った側は、createではなくinfoを引数として__init__を直接呼び出す。
#     <process 1>
#     tro = TransparentRemoteObject.create(cls,{"max_concurrency":16,"num_cpus":1.0},*args,**kwargs)
#     info = tro._get_info()
#     send(info) # 何らかの手段で送る。
#     <process 2>
#     info = receive() # 何らかの手段で受け取る。
#     tro = TransparentRemoteObject(info)
#
# ====== limitations ======
# <general>
# * "_get_info","unwrap","remote"という属性を持つオブジェクトを使用しないこと
# * 特殊メソッドのうち、デスクリプタ(__get__, __set__, __delete__)は未対応
# <numpy>
# * __array_function__ protocolには未対応
# * __array__()を呼ぶタイプの処理に対してはarrayのコピーが発生する
# <torch>
# * torch.no_grad()はスレッドローカルなコンテキストマネージャーであるため、これを使用するTransparentRemoteObjectはmax_concurrencyを1としておく必要がある。
#
import threading
import collections
import uuid
import operator
import math
import ray
import numpy as np
try:
	import torch
	__torch_imported=True
except ImportError:
	__torch_imported=False

def is_immutable(x):
	if x is None:
		return True
	return isinstance(x,(bool,int,float,complex,str,range,bytes))

def unwrap_if_needed(arg):
	if isinstance(arg,TransparentRemoteObject):
		return arg.unwrap()
	else:
		if isinstance(arg,collections.abc.Mapping):
			return type(arg)({k:unwrap_if_needed(a) for k,a in arg.items()})
		elif isinstance(arg,str):
			return arg
		elif isinstance(arg,collections.abc.Sequence):
			return type(arg)([unwrap_if_needed(a) for a in arg])
		else:
			return arg

def map_r(arg,func,keepRemote=False):
	if isinstance(arg,TransparentRemoteObject):
		if keepRemote:
			return func(arg)
		else:
			return map_r(arg.unwrap(),func,keepRemote)
	if isinstance(arg,collections.abc.Mapping):
		return type(arg)({k:map_r(a,func,keepRemote) for k,a in arg.items()})
	elif isinstance(arg,str):
		return func(arg)
	elif isinstance(arg,collections.abc.Sequence):
		return type(arg)([map_r(a,func,keepRemote) for a in arg])
	else:
		return func(arg)

class TransparentRemoteObjectInfo:
	def __init__(self,name,namespace,attr_uuid=None):
		self.name=name
		self.namespace=namespace
		self.attr_uuid=attr_uuid
	def copy(self):
		return TransparentRemoteObjectInfo(self.name,self.namespace,self.attr_uuid)
	def is_on_the_same_actor(self,other):
		return self.name==other.name and self.namespace==other.namespace
	def get_actor(self):
		return ray.get_actor(self.name,self.namespace)
	def create_info_for(self,attr_uuid):
		return TransparentRemoteObjectInfo(self.name,self.namespace,attr_uuid)

class TransparentRemoteObjectAsyncRef:
	def __init__(self,tro,ref):
		self.__tro=tro
		self.__ref=ref
		self.__isValid=True
	def __call__(self):
		assert self.__isValid,"Do not call a remote ref twice!"
		immutability,ret=ray.get(self.__ref)
		self.__isValid=False
		self.__tro=None
		self.__ref=None
		if immutability:
			return ret
		else:
			return TransparentRemoteObject(ret)

@ray.remote
class TransparentRemoteObjectBG:
	def __init__(self,info,bgClass,*args,**kwargs):
		self.__lock=threading.RLock()
		self.__info=info.copy()
		self.obj=bgClass(*args,**kwargs)
		self.attrMap={}
		self.base_uuid=self.get_uuid()
		self.attrMap[self.base_uuid]=self.obj
		self.attrRefCount=collections.defaultdict(lambda:0)
	def clear(self):
		with self.__lock:
			self.attrMap={self.base_uuid:self.obj}
	def get_uuid(self):
		with self.__lock:
			ret=uuid.uuid4()
			while ret in self.attrMap:
				ret=uuid.uuid4()
			return ret
	def get_value(self,actor_info):
		actor_name,namespace,attr_uuid=actor_info
		if self.__actor_name==actor_name and self.__namespace==namespace:
			return self.attrMap[attr_uuid]
		else:
			actor=ray.get_actor(actor_name,namespace)
			return ray.get(actor.get_value.remote(actor_info))
	def inc_ref(self,attr_uuid):
		#print("<inc_ref>",attr_uuid,"</inc_ref>")
		with self.__lock:
			self.attrRefCount[attr_uuid]+=1
	def release(self,attr_uuid):
		#print("<release>",attr_uuid,"</release>")
		with self.__lock:
			if attr_uuid in self.attrRefCount:
				self.attrRefCount[attr_uuid]-=1
				if self.attrRefCount[attr_uuid]<=0:
					self.attrMap.pop(attr_uuid)
					self.attrRefCount.pop(attr_uuid)
	def get_class(self,attr_uuid):
		return self.attrMap[attr_uuid].__class__
	def get_base_uuid(self):
		return self.base_uuid
	def unwrap(self,attr_uuid):
		return self.attrMap[attr_uuid]
	def __process_arg(self,arg):
		if isinstance(arg,TransparentRemoteObjectInfo):
			#TransparentRemoteObjectならTransparentRemoteObjectInfoで渡される
			if self.__info.is_on_the_same_actor(arg):
				#同じActorなら自身のattrMapから取得
				return self.attrMap[arg.attr_uuid]
			else:
				#異なるActorならTransparentRemoteObjectとして復元
				return TransparentRemoteObject(arg)
		else:
			if isinstance(arg,collections.abc.Mapping):
				return type(arg)({k:self.__process_arg(a) for k,a in arg.items()})
			elif isinstance(arg,str):
				return arg
			elif isinstance(arg,collections.abc.Sequence):
				return type(arg)([self.__process_arg(a) for a in arg])
			else:
				return arg
	def __process_args(self,*args,**kwargs):
		return tuple(self.__process_arg(a) for a in args),{k:self.__process_arg(v) for k,v in kwargs.items()}
	def __process_return(self,ret):
		if isinstance(ret,TransparentRemoteObject):
			return False, ret._get_info()
		elif is_immutable(ret):
			return True, ret
		else:
			new_uuid=self.get_uuid()
			self.attrMap[new_uuid]=ret
			return False, self.__info.create_info_for(new_uuid)
	#文字列表現
	def call_repr(self,attr_uuid):
		ret = repr(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def to_str(self,attr_uuid):
		ret = str(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def to_bytes(self,attr_uuid):
		ret = bytes(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_format(self,attr_uuid,format_spec):
		ret = format(self.attrMap[attr_uuid],format_spec)
		return self.__process_return(ret)
	#比較系
	def lt(self,attr_uuid,other):
		ret = self.attrMap[attr_uuid]<self.__process_arg(other)
		return self.__process_return(ret)
	def le(self,attr_uuid,other):
		ret = self.attrMap[attr_uuid]<=self.__process_arg(other)
		return self.__process_return(ret)
	def eq(self,attr_uuid,other):
		ret = self.attrMap[attr_uuid]==self.__process_arg(other)
		return self.__process_return(ret)
	def ne(self,attr_uuid,other):
		ret = self.attrMap[attr_uuid]!=self.__process_arg(other)
		return self.__process_return(ret)
	def gt(self,attr_uuid,other):
		ret = self.attrMap[attr_uuid]>self.__process_arg(other)
		return self.__process_return(ret)
	def ge(self,attr_uuid,other):
		ret = self.attrMap[attr_uuid]>=self.__process_arg(other)
		return self.__process_return(ret)
	#ハッシュ
	def call_hash(self,attr_uuid):
		ret = hash(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	#真理値
	def call_bool(self,attr_uuid):
		ret = bool(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	#属性
	def get_attr(self,attr_uuid,name):
		ret=getattr(self.attrMap[attr_uuid],name)
		return self.__process_return(ret)
	def set_attr(self,attr_uuid,name,value):
		original_type=type(value)
		value=self.__process_arg(value)
		def modifier(x):
			if isinstance(x,np.ndarray):
				return x.copy()
			else:
				return x
		value=map_r(value,modifier,True)
		setattr(self.attrMap[attr_uuid],name,value)
	def del_attr(self,attr_uuid,name):
		delattr(self.attrMap[attr_uuid],name)	
	def call_dir(self,attr_uuid):
		#敢えてimmutableと同様にlistそのものを返す
		return True, dir(self.attrMap[attr_uuid])
	#デスクリプタ
	'''
	def call_get(self,instance,owner=None):
		ret=self.attrMap[attr_uuid].__get__(instance,owner)
		return self.__process_return(ret)
	def call_set(self,instance,value):
		self.attrMap[attr_uuid].__set__(instance,value)
	def call_delete(self,instance):
		self.attrMap[attr_uuid].__delete__(instance)
	'''
	#メソッド呼び出し
	def call(self,attr_uuid,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		ret=self.attrMap[attr_uuid](*args,**kwargs)
		return self.__process_return(ret)
	def async_call(self,attr_uuid,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		#print("<async_call>",attr_uuid,"</async_call>")
		ret=self.attrMap[attr_uuid](*args,**kwargs)
		return self.__process_return(ret)
	#コンテナ
	def call_len(self,attr_uuid):
		ret = len(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_length_hint(self,attr_uuid):
		ret = operator.length_hint(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def get_item(self,attr_uuid,key):
		ret = self.attrMap[attr_uuid][key]
		return self.__process_return(ret)
	def set_item(self,attr_uuid,key,value):
		self.attrMap[attr_uuid][key]=self.__process_arg(value)
	def del_item(self,attr_uuid,key):
		del self.attrMap[attr_uuid][key]
	'''
	def missing(self,attr_uuid,key):
		#通常は__missing__単体で呼ばれることはないはず
		ret=self.attrMap[attr_uuid].__missing__(key)
		return self.__process_return(ret)
	'''
	def call_iter(self,attr_uuid):
		ret = iter(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_reversed(self,attr_uuid):
		ret = reversed(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_contains(self,attr_uuid,item):
		ret = self.__process_arg(item) in self.attrMap[attr_uuid]
		return self.__process_return(ret)
	#ジェネレータ
	def call_next(self,attr_uuid):
		ret = next(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	#算術系
	def add(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]+self.__process_arg(other)
		return self.__process_return(ret)
	def sub(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]-self.__process_arg(other)
		return self.__process_return(ret)
	def mul(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]*self.__process_arg(other)
		return self.__process_return(ret)
	def matmul(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]@self.__process_arg(other)
		return self.__process_return(ret)
	def truediv(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]/self.__process_arg(other)
		return self.__process_return(ret)
	def floordiv(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]//self.__process_arg(other)
		return self.__process_return(ret)
	def mod(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]%self.__process_arg(other)
		return self.__process_return(ret)
	def call_divmod(self,attr_uuid,other):
		ret=divmod(self.attrMap[attr_uuid],self.__process_arg(other))
		return self.__process_return(ret)
	def call_pow(self,attr_uuid,other,modulo=None):
		if modulo is None:
			ret=pow(self.attrMap[attr_uuid],self.__process_arg(other))
			return self.__process_return(ret)
		else:
			ret=pow(self.attrMap[attr_uuid],self.__process_arg(other),self.__process_arg(modulo))
			return self.__process_return(ret)
	def lshift(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]<<self.__process_arg(other)
		return self.__process_return(ret)
	def rshift(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]>>self.__process_arg(other)
		return self.__process_return(ret)
	def call_and(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]&self.__process_arg(other)
		return self.__process_return(ret)
	def call_xor(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]^self.__process_arg(other)
		return self.__process_return(ret)
	def call_or(self,attr_uuid,other):
		ret=self.attrMap[attr_uuid]|self.__process_arg(other)
		return self.__process_return(ret)
	def radd(self,attr_uuid,other):
		ret=self.__process_arg(other)+self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rsub(self,attr_uuid,other):
		ret=self.__process_arg(other)-self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rmul(self,attr_uuid,other):
		ret=self.__process_arg(other)*self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rmatmul(self,attr_uuid,other):
		ret=self.__process_arg(other)@self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rtruediv(self,attr_uuid,other):
		ret=self.__process_arg(other)/self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rfloordiv(self,attr_uuid,other):
		ret=self.__process_arg(other)//self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rmod(self,attr_uuid,other):
		ret=self.__process_arg(other)%self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rdivmod(self,attr_uuid,other):
		ret=divmod(self.__process_arg(other),self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def rpow(self,attr_uuid,other):
		ret=self.__process_arg(other)**self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rlshift(self,attr_uuid,other):
		ret=self.__process_arg(other)<<self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rrshift(self,attr_uuid,other):
		ret=self.__process_arg(other)>>self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rand(self,attr_uuid,other):
		ret=self.__process_arg(other)&self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def rxor(self,attr_uuid,other):
		ret=self.__process_arg(other)^self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def ror(self,attr_uuid,other):
		ret=self.__process_arg(other)|self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def iadd(self,attr_uuid,other):
		self.attrMap[attr_uuid]+=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def isub(self,attr_uuid,other):
		self.attrMap[attr_uuid]-=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def imul(self,attr_uuid,other):
		self.attrMap[attr_uuid]*=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def imatmul(self,attr_uuid,other):
		self.attrMap[attr_uuid]@=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def itruediv(self,attr_uuid,other):
		self.attrMap[attr_uuid]/=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def ifloordiv(self,attr_uuid,other):
		self.attrMap[attr_uuid]//=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def imod(self,attr_uuid,other):
		self.attrMap[attr_uuid]%=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def ipow(self,attr_uuid,other):
		self.attrMap[attr_uuid]**=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def ilshift(self,attr_uuid,other):
		self.attrMap[attr_uuid]<<=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def irshift(self,attr_uuid,other):
		self.attrMap[attr_uuid]>>=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def iand(self,attr_uuid,other):
		self.attrMap[attr_uuid]&=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def ixor(self,attr_uuid,other):
		self.attrMap[attr_uuid]^=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def ior(self,attr_uuid,other):
		self.attrMap[attr_uuid]|=self.__process_arg(other)
		return False, self.__info.create_info_for(attr_uuid)
	def neg(self,attr_uuid):
		ret=-self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def pos(self,attr_uuid):
		ret=+self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def call_abs(self,attr_uuid):
		ret=abs(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def invert(self,attr_uuid):
		ret=~self.attrMap[attr_uuid]
		return self.__process_return(ret)
	def to_complex(self,attr_uuid):
		ret = complex(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def to_int(self,attr_uuid):
		ret = int(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def to_float(self,attr_uuid):
		ret = float(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def to_index(self,attr_uuid):
		ret = operator.index(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_round(self,attr_uuid,ndigits=None):
		if ndigits is None:
			ret = round(self.attrMap[attr_uuid])
			return self.__process_return(ret)
		else:
			ret = round(self.attrMap[attr_uuid],self.__process_arg(ndigits))
			return self.__process_return(ret)
	def call_trunc(self,attr_uuid):
		ret = math.trunc(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_floor(self,attr_uuid):
		ret = math.floor(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	def call_ceil(self,attr_uuid):
		ret = math.ceil(self.attrMap[attr_uuid])
		return self.__process_return(ret)
	#コンテキストマネージャ
	def call_enter(self,attr_uuid,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		ret=self.attrMap[attr_uuid].__enter__(*args,**kwargs)
		return self.__process_return(ret)
	def call_exit(self,attr_uuid,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		self.attrMap[attr_uuid].__exit__(*args,**kwargs)
	#numpy
	def call_array_ufunc(self,attr_uuid,ufunc,method,*inputs,**kwargs):
		#in-placeなufuncを実現するためのもの
		inputs,kwargs=self.__process_args(*inputs,**kwargs)
		ret=getattr(ufunc,method)(*inputs,**kwargs)
		return self.__process_return(ret)

class TransparentRemoteObject(object):
	@classmethod
	def create(cls,baseValueClass,options: collections.abc.Mapping,*args,**kwargs):
		namespace=ray.get_runtime_context().namespace
		def check_actor_name(name,namespace):
			try:
				ray.get_actor(name,namespace)
			except:
				return True
			return False
		actor_name=str(uuid.uuid4())
		while not check_actor_name(actor_name,namespace):
			actor_name=str(uuid.uuid4())
		info=TransparentRemoteObjectInfo(actor_name,namespace)
		options={k:v for k,v in options.items()}
		if not "max_concurrency" in options:
			options["max_concurrency"]=1
		actor=TransparentRemoteObjectBG.options(name=actor_name,namespace=namespace,**options).remote(info,baseValueClass,*args,**kwargs)
		return cls(info,actor)
	def __init__(self,info, actor=None):
		self.__info=info.copy()
		if actor is None:
			self.__actor=info.get_actor()
		else:
			self.__actor=actor
		if self.__info.attr_uuid is None:
			self.__info.attr_uuid=ray.get(self.__actor.get_base_uuid.remote())
		self.__attr_uuid=self.__info.attr_uuid
		ray.get(self.__actor.inc_ref.remote(self.__attr_uuid))
	def __getstate__(self):
		return self.__info
	def __setstate__(self,info):
		self.__init__(info)
	def __get_bg_class(self):
		return ray.get(self.__actor.get_class.remote(self.__attr_uuid))
	def _get_info(self):
		return self.__info.copy()
	def unwrap(self):
		return ray.get(self.__actor.unwrap.remote(self.__attr_uuid))
	def __process_arg(self,arg):
		if isinstance(arg,TransparentRemoteObject):
			#TransparentRemoteObjectならTransparentRemoteObjectInfoで渡す(ray.actorを直接渡せないため)
			return arg.__info
		else:
			if isinstance(arg,collections.abc.Mapping):
				return type(arg)({k:self.__process_arg(a) for k,a in arg.items()})
			elif isinstance(arg,str):
				return arg
			elif isinstance(arg,collections.abc.Sequence):
				return type(arg)([self.__process_arg(a) for a in arg])
			else:
				return arg
	def __process_args(self,*args,**kwargs):
		return tuple(self.__process_arg(a) for a in args),{k:self.__process_arg(v) for k,v in kwargs.items()}
	def __process_return(self,ref):
		return TransparentRemoteObjectAsyncRef(self,ref)()
	def __del__(self):
		#print("<__del__>",self.__attr_uuid,"</__del__>")
		if ray is not None and ray.is_initialized is not None and ray.is_initialized():
			ray.get(self.__actor.release.remote(self.__attr_uuid))
	#文字列表現
	def __repr__(self):
		ref=self.__actor.call_repr.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __str__(self):
		ref=self.__actor.to_str.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __bytes__(self):
		ref=self.__actor.to_bytes.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __format__(self,format_spec):
		ref=self.__actor.call_format.remote(self.__attr_uuid,format_spec)
		return self.__process_return(ref)
	#比較系
	def __lt__(self,other):
		ref=self.__actor.lt.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __le__(self,other):
		ref=self.__actor.le.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __eq__(self,other):
		ref=self.__actor.eq.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ne__(self,other):
		ref=self.__actor.ne.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __gt__(self,other):
		ref=self.__actor.gt.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ge__(self,other):
		ref=self.__actor.ge.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	#ハッシュ
	def __hash__(self):
		ref=self.__actor.call_hash.remote(self.__attr_uuid)
		return self.__process_return(ref)
	#真理値
	def __bool__(self):
		ref=self.__actor.call_bool.remote(self.__attr_uuid)
		return self.__process_return(ref)
	#属性
	def __getattribute__(self,name):
		if name == "__class__":
			return self.__get_bg_class()
		return object.__getattribute__(self,name)
	def __getattr__(self,name):
		if name in [
			"_TransparentRemoteObject__info",
			"_TransparentRemoteObject__actor",
			"_TransparentRemoteObject__attr_uuid",
			"_get_info",
			"unwrap",
			"remote",
			"_TransparentRemoteObject__process_arg",
			"_TransparentRemoteObject__process_args",
			"_TransparentRemoteObject__process_return"
		]:
			return object.__getattr__(self,name)
		if isinstance(self,np.ndarray):
			if name in ["__array_struct__", "__array_interface__"]:
				raise AttributeError
			if name in ["__array_prepare__"]:
				return getattr(self.unwrap(),name)
		if name in ["__array__"]:
			#Numpy用
			return getattr(self.unwrap(),name)
		ref=self.__actor.get_attr.remote(self.__attr_uuid,name)
		return self.__process_return(ref)
	def __setattr__(self,name,value):
		if name in [
			"__class__",
			"_TransparentRemoteObject__info",
			"_TransparentRemoteObject__actor",
			"_TransparentRemoteObject__attr_uuid",
			"_get_info",
			"unwrap",
			"remote",
			"_TransparentRemoteObject__process_arg",
			"_TransparentRemoteObject__process_args",
			"_TransparentRemoteObject__process_return"
		]:
			object.__setattr__(self,name,value)
			return
		ray.get(self.__actor.set_attr.remote(self.__attr_uuid,name,self.__process_arg(value)))
	def __delattr__(self,name):
		if name in [
			"__class__",
			"_TransparentRemoteObject__info",
			"_TransparentRemoteObject__actor",
			"_TransparentRemoteObject__attr_uuid",
			"_get_info",
			"unwrap",
			"remote",
			"_TransparentRemoteObject__process_arg",
			"_TransparentRemoteObject__process_args",
			"_TransparentRemoteObject__process_return"
		]:
			object.__delattr__(self,name)
			return
		ray.get(self.__actor.del_attr.remote(self.__attr_uuid,name))
	def __dir__(self):
		#敢えてimmutableと同様にdictそのものを返す
		ref=self.__actor.call_dir.remote(self.__attr_uuid)
		return self.__process_return(ref)
	#デスクリプタ
	'''
	def __get__(self,instance,owner=None):
		ref=self.__actor.call_get.remote(self.__attr_uuid,instance,owner)
		return self.__process_return(ref)
	def __set__(self,instance,value):
		ray.get(self.__actor.call_set.remote(self.__attr_uuid,instance,value))
	def __delete__(self,instance,value):
		ray.get(self.__actor.call_delete.remote(self.__attr_uuid,instance))
	'''
	#メソッド呼び出し
	def __call__(self,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		ref=self.__actor.call.remote(self.__attr_uuid,*args,**kwargs)
		return self.__process_return(ref)
	def remote(self,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		#print("<remote>",self.__attr_uuid,"</remote>")
		return TransparentRemoteObjectAsyncRef(
			self,
			self.__actor.async_call.remote(self.__attr_uuid,*args,**kwargs)
		)
	#コンテナ
	def __len__(self):
		ref=self.__actor.call_len.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __length_hint__(self):
		ref=self.__actor.call_length_hint.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __getitem__(self,index):
		ref=self.__actor.get_item.remote(self.__attr_uuid,index)
		return self.__process_return(ref)
	def __setitem__(self,key,value):
		ray.get(self.__actor.set_item.remote(self.__attr_uuid,key,self.__process_arg(value)))
	def __delitem__(self,key):
		ray.get(self.__actor.del_item.remote(self.__attr_uuid,key))
	"""
	def __missing__(self,key):
		#通常は__missing__単体で呼ばれることはないはず
		ref=self.__actor.missing.remote(self.__attr_uuid,key)
		return self.__process_return(ref)
	"""
	def __iter__(self):
		ref=self.__actor.call_iter.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __reversed__(self):
		ref=self.__actor.call_reversed.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __contains__(self,item):
		ref=self.__actor.call_contains.remote(self.__attr_uuid,self.__process_arg(item))
		return self.__process_return(ref)
	#ジェネレータ
	def __next__(self):
		ref=self.__actor.call_next.remote(self.__attr_uuid)
		return self.__process_return(ref)
	#算術系
	def __add__(self,other):
		ref=self.__actor.add.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __sub__(self,other):
		ref=self.__actor.sub.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __mul__(self,other):
		ref=self.__actor.mul.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __matmul__(self,other):
		ref=self.__actor.matmul.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __truediv__(self,other):
		ref=self.__actor.truediv.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __floordiv__(self,other):
		ref=self.__actor.floordiv.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __mod__(self,other):
		ref=self.__actor.mod.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __divmod__(self,other):
		ref=self.__actor.call_divmod.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __pow__(self,other,modulo=None):
		ref=self.__actor.call_pow.remote(self.__attr_uuid,self.__process_arg(other),self.__process_arg(modulo))
		return self.__process_return(ref)
	def __lshift__(self,other):
		ref=self.__actor.lshift.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rshift__(self,other):
		ref=self.__actor.rshift.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __and__(self,other):
		ref=self.__actor.call_and.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __xor__(self,other):
		ref=self.__actor.call_xor.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __or__(self,other):
		ref=self.__actor.call_or.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __radd__(self,other):
		ref=self.__actor.radd.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rsub__(self,other):
		ref=self.__actor.rsub.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rmul__(self,other):
		ref=self.__actor.rmul.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rmatmul__(self,other):
		ref=self.__actor.rmatmul.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rtruediv__(self,other):
		ref=self.__actor.rtruediv.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rfloordiv__(self,other):
		ref=self.__actor.rfloordiv.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rmod__(self,other):
		ref=self.__actor.rmod.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rdivmod__(self,other):
		ref=self.__actor.rdivmod.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rpow__(self,other):
		ref=self.__actor.rpow.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rlshift__(self,other):
		ref=self.__actor.rlshift.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rrshift__(self,other):
		ref=self.__actor.rrshift.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rand__(self,other):
		ref=self.__actor.rand.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __rxor__(self,other):
		ref=self.__actor.rxor.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ror__(self,other):
		ref=self.__actor.ror.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __iadd__(self,other):
		ref=self.__actor.iadd.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __isub__(self,other):
		ref=self.__actor.isub.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __imul__(self,other):
		ref=self.__actor.imul.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __imatmul__(self,other):
		ref=self.__actor.imatmul.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __itruediv__(self,other):
		ref=self.__actor.itruediv.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ifloordiv__(self,other):
		ref=self.__actor.ifloordiv.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __imod__(self,other):
		ref=self.__actor.imod.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ipow__(self,other):
		ref=self.__actor.ipow.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ilshift__(self,other):
		ref=self.__actor.ilshift.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __irshift__(self,other):
		ref=self.__actor.irshift.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __iand__(self,other):
		ref=self.__actor.iand.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ixor__(self,other):
		ref=self.__actor.ixor.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __ior__(self,other):
		ref=self.__actor.ior.remote(self.__attr_uuid,self.__process_arg(other))
		return self.__process_return(ref)
	def __neg__(self):
		ref=self.__actor.neg.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __pos__(self):
		ref=self.__actor.pos.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __abs__(self):
		ref=self.__actor.call_abs.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __invert__(self):
		ref=self.__actor.invert.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __complex__(self):
		ref=self.__actor.to_complex.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __int__(self):
		ref=self.__actor.to_int.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __float__(self):
		ref=self.__actor.to_float.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __index__(self):
		ref=self.__actor.to_index.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __round__(self,ndigits=None):
		ref=self.__actor.call_round.remote(self.__attr_uuid,self.__process_arg(ndigits))
		return self.__process_return(ref)
	def __trunc__(self):
		ref=self.__actor.call_trunc.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __floor__(self):
		ref=self.__actor.call_floor.remote(self.__attr_uuid)
		return self.__process_return(ref)
	def __ceil__(self):
		ref=self.__actor.call_ceil.remote(self.__attr_uuid)
		return self.__process_return(ref)
	#コンテキストマネージャ
	def __enter__(self,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		ref=self.__actor.call_enter.remote(self.__attr_uuid,*args,**kwargs)
		return self.__process_return(ref)
	def __exit__(self,*args,**kwargs):
		args,kwargs=self.__process_args(*args,**kwargs)
		ray.get(self.__actor.call_exit.remote(self.__attr_uuid,*args,**kwargs))
	#numpy
	def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
		#remoteにあるarrayに対するin-place operationを検出してactorに投げる
		#in-placeでない引数についてはunwrapしても差し支えないのでunwrapしてしまうことによって再帰を避ける
		def unwrapper(x):
			return x.unwrap() if isinstance(x,TransparentRemoteObject) else x

		if "out" in kwargs and isinstance(kwargs["out"][0],TransparentRemoteObject):
			#kwargs["out"]がある場合は基本的にこれに対してin-placeの処理になる
			#その場合、kwargs["out"]は1つ以上のndarrayが入ったtupleである
			if id(self)!=id(kwargs["out"][0]):
				return kwargs["out"][0].__array_ufunc__(ufunc,method,*inputs,**kwargs)
			else:
				inputs,kwargs=self.__process_args(*inputs,**kwargs)
				ref=self.__actor.call_array_ufunc.remote(self.__attr_uuid,ufunc,method,*inputs,**kwargs)
				return self.__process_return(ref)
		elif method=="at" and isinstance(inputs[0],TransparentRemoteObject):
			#atの場合はinputs[0]に対するin-placeの処理となる。
				if id(self)!=id(inputs[0]):
					return inputs[0].__array_ufunc__(ufunc,method,*inputs,**kwargs)
				else:
					inputs,kwargs=self.__process_args(*inputs,**kwargs)
					ref=self.__actor.call_array_ufunc.remote(self.__attr_uuid,ufunc,method,*inputs,**kwargs)
					return self.__process_return(ref)
		else:
			#in-placeな処理が行われない場合は、全てunwrapして通常のufunc処理に委ねる
			inputs=map_r(inputs,unwrapper)
			kwargs=map_r(kwargs,unwrapper)
			return getattr(ufunc,method)(*inputs,**kwargs)

#torch
if __torch_imported is not None:
	def __torch_function__(cls,func,types,args=(),kwargs=None):
		ext_args = tuple(a.unwrap() if isinstance(a,cls) else a for a in args)
		if kwargs is None:
			kwargs = {}
		ret=func(*ext_args,**kwargs)
		for aft,bef in zip(ext_args,args):
			if isinstance(bef,torch.Tensor) and isinstance(bef,cls):
				if (aft!=bef.unwrap()).any():
					bef[:]=aft[:]
		return ret
	bound=__torch_function__.__get__(TransparentRemoteObject,type(TransparentRemoteObject))
	setattr(TransparentRemoteObject,"__torch_function__",bound)

# ============
# テストコード
# ============

class DummyForMatmul:
	def __init__(self,v):
		self.v=v
	def __eq__(self,other):
		return self.v==other.v
	def __str__(self):
		return "DummyForMatmul("+str(self.v)+")"
	def __matmul__(self,other):
		return DummyForMatmul(self.v*other.v)
	def __rmatmul__(self,other):
		return DummyForMatmul(other.v*self.v)
	def __imatmul__(self,other):
		self.v*=other.v
		return self
class TestBGObject:
	def __init__(self):
		self.v_bool=False
		self.v_int=-100
		self.v_float=12.345
		self.v_str="string"
		self.v_list=[0,1.0,"a",None]
		self.v_dict={"a":None,"b":"B","c":[4,5.0]}
		self.v_np_1darray=np.array([1,2,3])
		self.v_np_2darray=np.array([[1.0,0.0,0.0],[0.0,0.0,2.0],[0.0,3.0,0.0]])
		self.v_dummy_for_matmul=DummyForMatmul(3.0)
	def func_no_arg(self):
		return "Hello world"
	def func_two_args(self,x,y):
		return x+y
	def func_with_mutable_args(self,x,*,y):
		return len(x)+len(y)
	def __repr__(self):
		return "__repr__: TestBGObject()"
	def __str__(self):
		return "__str__: TestBGObject()"
	def __bytes__(self):
		return "__bytes__: TestBGObject()".encode()
	def __format__(self,format_spec):
		return "__format__: "+format_spec
	def __hash__(self):
		return 1234567890
	def __bool__(self):
		return False
	def __lt__(self,other):
		return self.v_float<other
	def __le__(self,other):
		return self.v_float<=other
	def __eq__(self,other):
		return self.v_float==other
	def __ne__(self,other):
		return self.v_float!=other
	def __gt__(self,other):
		return self.v_float>other
	def __ge__(self,other):
		return self.v_float>=other
	def __add__(self,other):
		return self.v_float+other
	def __sub__(self,other):
		return self.v_float-other
	def __mul__(self,other):
		return self.v_float*other
	def __truediv__(self,other):
		return self.v_float/other
	def __floordiv__(self,other):
		return self.v_float//other
	def __mod__(self,other):
		return self.v_int%other
	def __divmod__(self,other):
		return divmod(self.v_int,other)
	def __pow__(self,other,modulo=None):
		return self.v_float**other
	def __lshift__(self,other):
		return self.v_int<<other
	def __rshift__(self,other):
		return self.v_int>>other
	def __and__(self,other):
		return self.v_int&other
	def __xor__(self,other):
		return self.v_int^other
	def __or__(self,other):
		return self.v_int|other
	def __radd__(self,other):
		return other+self.v_float
	def __rsub__(self,other):
		return other-self.v_float
	def __rmul__(self,other):
		return other*self.v_float
	def __rtruediv__(self,other):
		return other/self.v_float
	def __rfloordiv__(self,other):
		return other//self.v_float
	def __rmod__(self,other):
		return other%self.v_int
	def __rdivmod__(self,other):
		return divmod(other,self.v_int)
	def __rpow__(self,other,modulo=None):
		return other**self.v_float
	def __rlshift__(self,other):
		return other<<self.v_int
	def __rrshift__(self,other):
		return other>>self.v_int
	def __rand__(self,other):
		return other&self.v_int
	def __rxor__(self,other):
		return other^self.v_int
	def __ror__(self,other):
		return other|self.v_int
	def __iadd__(self,other):
		self.v_float+=other
		return self
	def __isub__(self,other):
		self.v_float-=other
		return self
	def __imul__(self,other):
		self.v_float*=other
		return self
	def __itruediv__(self,other):
		self.v_float/=other
		return self
	def __ifloordiv__(self,other):
		self.v_float//=other
		return self
	def __imod__(self,other):
		self.v_int%=other
		return self
	def __ipow__(self,other):
		self.v_float**=other
		return self
	def __ilshift__(self,other):
		self.v_int<<=other
		return self
	def __irshift__(self,other):
		self.v_int>>=other
		return self
	def __iand__(self,other):
		self.v_int&=other
		return self
	def __ixor__(self,other):
		self.v_int^=other
		return self
	def __ior__(self,other):
		self.v_int|=other
		return self
	def __neg__(self):
		return -self.v_int
	def __pos__(self):
		return +self.v_int
	def __abs__(self):
		return abs(self.v_int)
	def __invert__(self):
		return ~self.v_int
	def __complex__(self):
		return complex(self.v_int)
	def __int__(self):
		return int(self.v_int)
	def __float__(self):
		return float(self.v_int)
	def __index__(self):
		return operator.index(self.v_int)
	def __round__(self,ndigits=None):
		return round(self.v_int,ndigits)
	def __trunc__(self):
		return math.trunc(self.v_int)
	def __floor__(self):
		return math.floor(self.v_int)
	def __ceil__(self):
		return math.ceil(self.v_int)

def checkGenericFunc(descriptor,bool_func):
	try:
		assert(bool_func())
		if verbose:
			print(descriptor,": success!")
	except Exception as e:
		if verbose:
			print(descriptor,": failure!")
		raise e
def checkReturnValue(descriptor,tro,fro,func,expected=None):
	t=func(tro)
	f=func(fro)
	try:
		assert(t==f)
	except Exception as e:
		if verbose:
			print(descriptor,": failure! Different value: tro=",t,",fro=",f)
		raise e
	if expected is not None:
		try:
			assert(t==expected)
		except Exception as e:
			if verbose:
				print(descriptor,": failure! Unexpected value: tro=",t,",expected=",expected)
			raise e
	if verbose:
		print(descriptor,": success!")
def checkSetattr(descriptor,tro,fro,attr_name,value):
	try:
		setattr(tro,attr_name,value)
		setattr(fro,attr_name,value)
	except Exception as e:
		if verbose:
			print(descriptor,": failure!")
		raise e
	checkReturnValue(descriptor,tro,fro,lambda x:getattr(x,attr_name),value)
def checkDelattr(descriptor,tro,fro,attr_name):
	try:
		assert(hasattr(tro,attr_name))
		assert(hasattr(fro,attr_name))
		delattr(tro,attr_name)
		delattr(fro,attr_name)
		assert(not hasattr(tro,attr_name))
		assert(not hasattr(fro,attr_name))
		if verbose:
			print(descriptor,": success!")
	except Exception as e:
		if verbose:
			print(descriptor,": failure!")
		raise e
def checkSetitem(descriptor,tro,fro,key,value):
	try:
		tro[key]=value
		fro[key]=value
	except Exception as e:
		if verbose:
			print(descriptor,": failure!")
		raise e
	checkReturnValue(descriptor,tro,fro,lambda x:x[key],value)
def checkDelitem(descriptor,tro,fro,key):
	try:
		del tro[key]
		del fro[key]
		if verbose:
			print(descriptor,": success!")
	except Exception as e:
		if verbose:
			print(descriptor,": failure!")
		raise e
def checkBinaryOperator(opName,tro,fro,base,other,op,iop=None):
	checkReturnValue(opName,tro,fro,lambda x:op(x,other),op(base(tro),other))
	checkReturnValue("r"+opName,tro,fro,lambda x:op(other,x),op(other,base(tro)))
	if iop is not None:
		try:
			v=op(base(tro),other)
			iop(tro,other)
			iop(fro,other)
			assert(base(tro)==base(fro))
			assert(base(tro)==v)
			if verbose:
				print("i"+opName,": success!")
		except Exception as e:
			if verbose:
				print("i"+opName,": failure!")
				print(base(tro),",",base(fro),",",v)
			raise e
def checkUnaryOperator(opName,tro,fro,base,op):
	checkReturnValue(opName,tro,fro,lambda x:op(x),op(base(tro)))

if __name__=='__main__':
	verbose=True
	if not ray.is_initialized():
		ray.init()
	#ユニットテスト
	if verbose:
		print("========== Base unit test started. ==========")
	tro=TransparentRemoteObject.create(TestBGObject,{"max_concurrency":16,"num_cpus":1.0})
	tro2=TransparentRemoteObject.create(TestBGObject,{"max_concurrency":16,"num_cpus":1.0})
	fro=TestBGObject()

	#isinstanceの偽装
	checkGenericFunc("type",lambda:type(tro)==TransparentRemoteObject)
	checkGenericFunc("__class__",lambda:tro.__class__==TestBGObject)
	checkGenericFunc("isinstance (Front)",lambda:isinstance(tro,TransparentRemoteObject))
	checkGenericFunc("isinstance (Back)",lambda:isinstance(tro,TestBGObject))
	#文字列表現
	checkReturnValue("__repr__",tro,fro,lambda x:x.__repr__(),"__repr__: TestBGObject()")
	checkReturnValue("__str__",tro,fro,str,"__str__: TestBGObject()")
	checkReturnValue("__bytes__",tro,fro,bytes,"__bytes__: TestBGObject()".encode())
	checkReturnValue("__format__",tro,fro,lambda x:format(x,"+04e"),"__format__: +04e")
	#比較
	checkReturnValue("__lt__",tro,fro,lambda x:x<12.345,False)
	checkReturnValue("__le__",tro,fro,lambda x:x<=12.345,True)
	checkReturnValue("__eq__",tro,fro,lambda x:x==12.345,True)
	checkReturnValue("__ne__",tro,fro,lambda x:x!=12.345,False)
	checkReturnValue("__gt__",tro,fro,lambda x:x>12.345,False)
	checkReturnValue("__ge__",tro,fro,lambda x:x>=12.345,True)
	checkReturnValue("__lt__ (2 bgs mixed)",tro,fro,lambda x:x<tro2,False)
	#ハッシュ
	checkReturnValue("__hash__",tro,fro,hash,1234567890)
	#真理値
	checkReturnValue("__bool__",tro,fro,bool,False)
	#属性取得
	checkReturnValue("immutable getattr",tro,fro,lambda x:x.v_str,"string")
	checkReturnValue("mutable getattr",tro,fro,lambda x:x.v_list,[0,1.0,"a",None])
	checkSetattr("immutable setattr",tro,fro,"new_immutable_attr",-1.0)
	checkSetattr("mutable setattr",tro,fro,"new_mutable_attr",[-1.0,False,"str"])
	checkDelattr("immutable delattr",tro,fro,"new_immutable_attr")
	checkDelattr("mutable delattr",tro,fro,"new_mutable_attr")
	checkReturnValue("dir",tro,fro,lambda x:dir(x))
	checkSetattr("TransparentRemoteObject as an attribute of another",tro,fro,"new_remote_attr",tro2)
	def f():
		tro.v_shared_mutable=[67,89,10]
		tro2.v_shared_mutable=tro.v_shared_mutable
		result = tro.v_shared_mutable==tro2.v_shared_mutable
		tro2.v_shared_mutable[2]=-1
		result = result and tro.v_shared_mutable==tro2.v_shared_mutable
		return result
	checkGenericFunc("Sharing one TransparentRemoteObject by two actors",f)
	
	#デスクリプタ 未対応
	#メソッド呼び出し
	checkReturnValue("call method (w/o arg)",tro,fro,lambda x:x.func_no_arg(),"Hello world")
	checkReturnValue("call method (with immutable args)",tro,fro,lambda x:x.func_two_args(3,4),7)
	checkReturnValue("call method (with mutable args)",tro,fro,lambda x:x.func_with_mutable_args(tro.v_list,y=tro.v_dict),7)
	def f():
		ref1=tro.func_with_mutable_args.remote(tro.v_list,y=tro.v_dict)
		ref2=tro.func_two_args.remote(4,2)
		result = ref1()+ref2()==13
		return result
	checkGenericFunc("call methods asynchronously",f)
	#コンテナ
	checkReturnValue("len (list)",tro,fro,lambda x:len(x.v_list),4)
	checkReturnValue("len (dict)",tro,fro,lambda x:len(x.v_dict),3)
	checkReturnValue("length_hint (list)",tro.v_list,fro.v_list,
		lambda x:operator.length_hint(x),
		operator.length_hint([0,1.0,"a",None]))
	checkReturnValue("length_hint (dict)",tro.v_dict,fro.v_dict,
		lambda x:operator.length_hint(x),
		operator.length_hint({"a":None,"b":"B","c":[4,5.0]}))
	checkReturnValue("getitem (list)",tro,fro,lambda x:[x.v_list[0]]+x.v_list[1:],[0,1.0,"a",None])
	checkReturnValue("getitem (dict)",tro,fro,
		lambda x:{k:x.v_dict[k] for k in ["a","b","c"]},
		{"a":None,"b":"B","c":[4,5.0]})
	checkSetitem("setitem (list,replace)",tro.v_list,fro.v_list,slice(1,3),[44,99])
	checkSetitem("setitem (dict,replace)",tro.v_dict,fro.v_dict,"a",555)
	checkSetitem("setitem (dict,append)",tro.v_dict,fro.v_dict,"d",True)
	checkDelitem("delitem (list)",tro.v_list,fro.v_list,0)
	checkDelitem("delitem (dict)",tro.v_dict,fro.v_dict,"c")
	assert(tro.v_list==[44,99,None])
	assert(tro.v_dict=={"a":555,"b":"B","d":True})
	def f():
		result = [v for v in tro.v_list]==tro.v_list
		return result
	checkGenericFunc("iter & next (list)",f)
	def f():
		result = [v for v in reversed(tro.v_list)]==[tro.v_list[i] for i in reversed(range(len(tro.v_list)))]
		return result
	checkGenericFunc("reversed (list)",f)
	def f():
		result = {k:v for k,v in tro.v_dict.items()}==tro.v_dict
		return result
	checkGenericFunc("iter & next (dict)",f)
	def f():
		result = None in tro.v_list
		result = result and not 0 in fro.v_list
		return result
	checkGenericFunc("contains (list)",f)
	def f():
		result = "a" in tro.v_dict
		result = result and not "c" in tro.v_dict
		return result
	checkGenericFunc("contains (dict)",f)
	def gen():
		for i in range(10):
			yield i
		return
	def f():
		tro.gen=gen
		return (np.array([v for v in tro.gen()])==np.array(list(range(10)))).all()
	checkGenericFunc("generator",f)
	#算術系
	def iadd(x,y):x+=y
	checkBinaryOperator("add",tro,fro,lambda x:x.v_float,3,lambda x,y:x+y,iadd)
	def isub(x,y):x-=y
	checkBinaryOperator("sub",tro,fro,lambda x:x.v_float,3,lambda x,y:x-y,isub)
	def imul(x,y):x*=y
	checkBinaryOperator("mul",tro,fro,lambda x:x.v_float,3,lambda x,y:x*y,imul)
	def imatmul(x,y):x@=y
	checkBinaryOperator("matmul (dummy)",tro.v_dummy_for_matmul,fro.v_dummy_for_matmul,lambda x:x,DummyForMatmul(5.0),lambda x,y:x@y,imatmul)
	def itruediv(x,y):x/=y
	checkBinaryOperator("truediv",tro,fro,lambda x:x.v_float,3,lambda x,y:x/y,itruediv)
	tro.v_float=12.345
	fro.v_float=12.345
	def ifloordiv(x,y):x//=y
	checkBinaryOperator("floordiv",tro,fro,lambda x:x.v_float,3,lambda x,y:x//y,ifloordiv)
	tro.v_int=5
	fro.v_int=5
	def imod(x,y):x%=y
	checkBinaryOperator("mod",tro,fro,lambda x:x.v_int,3,lambda x,y:x%y,imod)
	tro.v_int=5
	fro.v_int=5
	checkBinaryOperator("divmod",tro,fro,lambda x:x.v_int,3,lambda x,y:divmod(x,y))
	def ipow(x,y):x**=y
	checkBinaryOperator("pow",tro,fro,lambda x:x.v_float,3,lambda x,y:x**y,ipow)
	tro.v_int=5
	fro.v_int=5
	def ilshift(x,y):x<<=y
	checkBinaryOperator("lshift",tro,fro,lambda x:x.v_int,3,lambda x,y:x<<y,ilshift)
	def irshift(x,y):x>>=y
	checkBinaryOperator("rshift",tro,fro,lambda x:x.v_int,3,lambda x,y:x>>y,irshift)
	def iand(x,y):x&=y
	checkBinaryOperator("and",tro,fro,lambda x:x.v_int,3,lambda x,y:x&y,iand)
	def ixor(x,y):x^=y
	checkBinaryOperator("xor",tro,fro,lambda x:x.v_int,3,lambda x,y:x^y,ixor)
	def ior(x,y):x|=y
	checkBinaryOperator("or",tro,fro,lambda x:x.v_int,3,lambda x,y:x|y,ior)
	tro.v_int=-5
	fro.v_int=-5
	checkUnaryOperator("neg",tro,fro,lambda x:x.v_int,lambda x:-x)
	checkUnaryOperator("pos",tro,fro,lambda x:x.v_int,lambda x:+x)
	checkUnaryOperator("abs",tro,fro,lambda x:x.v_int,lambda x:abs(x))
	checkUnaryOperator("invert",tro,fro,lambda x:x.v_int,lambda x:~x)
	checkUnaryOperator("complex",tro,fro,lambda x:x.v_int,lambda x:complex(x))
	checkUnaryOperator("int",tro,fro,lambda x:x.v_int,lambda x:int(x))
	checkUnaryOperator("float",tro,fro,lambda x:x.v_int,lambda x:float(x))
	checkUnaryOperator("index",tro,fro,lambda x:x.v_int,lambda x:operator.index(x))
	checkUnaryOperator("round",tro,fro,lambda x:x.v_int,lambda x:round(x,3))
	checkUnaryOperator("trunc",tro,fro,lambda x:x.v_int,lambda x:math.trunc(x))
	checkUnaryOperator("floor",tro,fro,lambda x:x.v_int,lambda x:math.floor(x))
	checkUnaryOperator("ceil",tro,fro,lambda x:x.v_int,lambda x:math.ceil(x))

	if verbose:
		print("========== Base unit test succeeded! ==========")

	#numpy test
	if verbose:
		print("========== Additional test for Numpy started. ==========")
	m1=np.array([-3.,-2.,1.])
	m2=np.array([[1.0,0.0,0.0],[0.0,0.0,2.0],[0.0,3.0,0.0]])
	def f():
		tro.v_np_1darray=m1
		fro.v_np_1darray=m1
		tro.v_np_2darray=m2
		return (tro.v_np_1darray==m1).all() and (tro.v_np_2darray==m2).all() and id(tro.v_np_1darray)!=id(m1) and id(tro.v_np_2darray)!=id(m2)
	checkGenericFunc("setattr",f)
	checkBinaryOperator("matmul",tro.v_np_1darray,fro.v_np_1darray,lambda x:x,np.array([2,2,3]),lambda x,y:float(x@y))
	#numpyはimatmul(In-placeなmatmul)をサポートしていないので省略
	def f():
		tro.v_np_1darray[0:2]=np.array([-1.,1.])
		m1[0:2]=np.array([-1.,1.])
		tro.v_np_2darray[0:2,1:3]=np.array([[4.,5.,],[-3.,-2.]])
		m2[0:2,1:3]=np.array([[4.,5.,],[-3.,-2.]])
		return (tro.v_np_1darray==m1).all() and (tro.v_np_2darray==m2).all()
	checkGenericFunc("assign by slice",f)
	def f():
		v=tro.v_np_2darray.view().reshape([9,1])
		v[5]=-7.0
		m2.view().reshape([9,1])[5]=-7.0
		return v.shape==(9,1) and tro.v_np_2darray.shape==(3,3) and (tro.v_np_2darray==m2).all()
	checkGenericFunc("view",f)
	def f():
		result = (np.exp(tro.v_np_2darray)==np.exp(m2)).all()
		return result
	checkGenericFunc("ufunc (call)",f)
	def f():
		#fresh result
		o1=np.add.reduce(tro.v_np_2darray)
		o2=np.add.reduce(m2)
		result = (o1==o2).all()
		#in-place
		o1=np.zeros(3)
		o2=np.zeros(3)
		tro.o1=np.zeros(3)
		tro.o2=np.zeros(3)
		np.add.reduce(tro.v_np_2darray,out=o1)
		np.add.reduce(tro.v_np_2darray,out=tro.o1)
		np.add.reduce(m2,out=o2)
		np.add.reduce(m2,out=tro.o2)
		result = result and (o1==o2).all() and (tro.o1==o2).all() and (o1==tro.o2).all()
		#in-place with "where"
		o1=np.zeros(3)
		o2=np.zeros(3)
		tro.o1=np.zeros(3)
		tro.o2=np.zeros(3)
		np.add.reduce(tro.v_np_2darray,out=o1,where=tro.v_np_2darray>0)
		np.add.reduce(tro.v_np_2darray,out=tro.o1,where=m2>0)
		np.add.reduce(m2,out=o2,where=tro.v_np_2darray>0)
		np.add.reduce(m2,out=tro.o2,where=m2>0)
		result = result and (o1==o2).all() and (tro.o1==o2).all() and (o1==tro.o2).all()
		return result
	checkGenericFunc("ufunc (reduce)",f)
	def f():
		tro.ind=[0,2,1,0]
		ind=[0,2,1,0]
		#fresh result
		o1=np.add.reduceat(tro.v_np_2darray,tro.ind)
		o2=np.add.reduceat(m2,tro.ind)
		result = (o1==o2).all()
		#in-place
		o1=np.zeros([4,3])
		o2=np.zeros([4,3])
		tro.o1=np.zeros([4,3])
		tro.o2=np.zeros([4,3])
		np.add.reduceat(tro.v_np_2darray,tro.ind,out=o1)
		np.add.reduceat(tro.v_np_2darray,tro.ind,out=tro.o1)
		np.add.reduceat(m2,tro.ind,out=o2)
		np.add.reduceat(m2,tro.ind,out=tro.o2)
		result = result and (o1==o2).all() and (tro.o1==o2).all() and (o1==tro.o2).all()
		return result
	checkGenericFunc("ufunc (reduceat)",f)
	def f():
		#fresh result
		o1=np.add.accumulate(tro.v_np_2darray)
		o2=np.add.accumulate(m2)
		result = (o1==o2).all()
		#in-place
		o1=np.zeros([3,3])
		o2=np.zeros([3,3])
		tro.o1=np.zeros([3,3])
		tro.o2=np.zeros([3,3])
		np.add.accumulate(tro.v_np_2darray,out=o1)
		np.add.accumulate(tro.v_np_2darray,out=tro.o1)
		np.add.accumulate(m2,out=o2)
		np.add.accumulate(m2,out=tro.o2)
		result = result and (o1==o2).all() and (tro.o1==o2).all() and (o1==tro.o2).all()
		return result
	checkGenericFunc("ufunc (accumulate)",f)
	def f():
		#fresh result
		o1=np.multiply.outer(tro.v_np_1darray,m1)
		o2=np.multiply.outer(m1,tro.v_np_1darray)
		o3=np.multiply.outer(tro.v_np_1darray,tro.v_np_1darray)
		o4=np.multiply.outer(m1,m1)
		result = (o1==o2).all() and (o1==o3).all() and (o1==o4).all()
		#in-place
		o1=np.zeros([3,3])
		o2=np.zeros([3,3])
		tro.o1=np.zeros([3,3])
		tro.o2=np.zeros([3,3])
		np.multiply.outer(m1,tro.v_np_1darray,out=o1)
		np.multiply.outer(tro.v_np_1darray,m1,out=tro.o1)
		np.multiply.outer(tro.v_np_1darray,tro.v_np_1darray,out=o2)
		np.multiply.outer(m1,m1,out=tro.o2)
		result = result and (o1==o2).all() and (tro.o1==o2).all() and (o1==tro.o2).all()
		return result
	checkGenericFunc("ufunc (outer)",f)
	def f():
		np.add.at(tro.v_np_1darray,[0,1],1)
		np.add.at(m1,[0,1],1)
		result = (tro.v_np_1darray==m1).all()
		return result
	checkGenericFunc("ufunc (at)",f)
	def f():
		def sub(x):
			return x*2+1
		vsub=np.vectorize(sub)
		result = (vsub(tro.v_np_2darray)==vsub(m2)).all()
		return result
	checkGenericFunc("vectorize",f)
	def f():
		result = np.linalg.norm(tro.v_np_1darray)==np.linalg.norm(m1)
		result = result and (np.linalg.inv(tro.v_np_2darray)==np.linalg.inv(m2)).all()
		return result
	checkGenericFunc("linalg (norm,inv)",f)
	def f():
		result = (np.dot(m2,tro.v_np_1darray)==np.dot(tro.v_np_2darray,m1)).all()
		result = result and (np.dot(tro.v_np_2darray,tro.v_np_1darray)==np.dot(m2,m1)).all()
		v=np.array([0.,5.,6.])
		result = result and (np.cross(v,tro.v_np_1darray)==np.cross(v,m1)).all()
		return result
	checkGenericFunc("dot and cross",f)

	if verbose:
		print("========== Additional test for Numpy succeeded. ==========")
	#torch test
	if __torch_imported is not None:
		if verbose:
			print("========== Additional test for PyTorch started. ==========")
		t1=torch.tensor([-3.,-2.,1.])
		t2=torch.tensor([[1.0,0.0,0.0],[0.0,0.0,2.0],[0.0,3.0,0.0]])
		def f():
			tro.v_torch_1d_tensor=t1
			tro.v_torch_2d_tensor=t2
			tro2.v_torch_1d_tensor=t1
			tro2.v_torch_2d_tensor=t2
			fro.v_torch_1d_tensor=t1
			return (tro.v_torch_1d_tensor==t1).all() and id(tro.v_torch_1d_tensor)!=id(t1)
		checkGenericFunc("setattr",f)
		checkBinaryOperator("matmul",tro.v_torch_1d_tensor,fro.v_torch_1d_tensor,lambda x:x,t2,lambda x,y:torch.sum(x@y))
		def f():
			v=tro.v_torch_1d_tensor@t2
			tro.v_torch_1d_tensor@=t2
			fro.v_torch_1d_tensor@=t2
			result = (tro.v_torch_1d_tensor==fro.v_torch_1d_tensor).all()
			result = result and (tro.v_torch_1d_tensor==v).all()
			tro.v_torch_1d_tensor=t1
			fro.v_torch_1d_tensor=t1
			return result
		checkGenericFunc("imatmul",f)
		def f():
			o1=tro.v_torch_1d_tensor.numpy()
			o2=t1.numpy()
			return (o1==o2).all()
		checkGenericFunc("tensor.numpy()",f)
		def f():
			tro.v_torch_1d_tensor[0:2]=torch.tensor([-1.,1.])
			t1[0:2]=torch.tensor([-1.,1.])
			tro.v_torch_2d_tensor[0:2,1:3]=torch.tensor([[4.,5.,],[-3.,-2.]])
			t2[0:2,1:3]=torch.tensor([[4.,5.,],[-3.,-2.]])
			return (tro.v_torch_1d_tensor==t1).all() and (tro.v_torch_2d_tensor==t2).all()
		checkGenericFunc("assign by slice",f)
		def f():
			v=tro.v_torch_2d_tensor.view([9,1])
			v[5]=-7.0
			t2.view([9,1])[5]=-7.0
			return v.shape==(9,1) and tro.v_torch_2d_tensor.shape==(3,3) and (tro.v_torch_2d_tensor==t2).all()
		checkGenericFunc("view",f)
		def f():
			o1=torch.sigmoid(tro.v_torch_1d_tensor)
			torch.sigmoid_(tro.v_torch_1d_tensor)
			result = (o1==tro.v_torch_1d_tensor).all()
			tro.v_torch_1d_tensor=t1
			return result
		checkGenericFunc("in-place operation",f)
		tro.v_torch_1d_tensor=t1
		tro2.v_torch_1d_tensor=t1
		def f():
			fc=torch.nn.Linear(3,1)
			y1=fc(tro.v_torch_1d_tensor)
			y1.backward()
			g1=[np.array(p.grad) for p in fc.parameters()]
			fc.zero_grad()
			y2=2*fc(tro2.v_torch_1d_tensor)
			y2.backward()
			g2=[np.array(p.grad) for p in fc.parameters()]
			fc.zero_grad()
			y3=3*fc(t1)
			y3.backward()
			g3=[np.array(p.grad) for p in fc.parameters()]
			result = np.array([(gg1*2==gg2).all() and (gg1*3==gg3).all() for gg1,gg2,gg3 in zip(g1,g2,g3)]).all()
			return result
		checkGenericFunc("nn.module with a remote tensor input",f)
		def f():
			tro.v_fc=torch.nn.Linear(3,1)
			y1=tro.v_fc(tro.v_torch_1d_tensor)
			y1.backward()
			g1=[np.array(p.grad) for p in tro.v_fc.parameters()]
			tro.v_fc.zero_grad()
			y2=2*tro.v_fc(tro2.v_torch_1d_tensor)
			y2.backward()
			g2=[np.array(p.grad) for p in tro.v_fc.parameters()]
			tro.v_fc.zero_grad()
			y3=3*tro.v_fc(t1)
			y3.backward()
			g3=[np.array(p.grad) for p in tro.v_fc.parameters()]
			result = np.array([(gg1*2==gg2).all() and (gg1*3==gg3).all() for gg1,gg2,gg3 in zip(g1,g2,g3)]).all()
			return result
		checkGenericFunc("remote nn.module",f)
		tro3=TransparentRemoteObject.create(TestBGObject,{"max_concurrency":1,"num_cpus":1.0}) #torch.no_grad()がスレッドローカルなため、max_concurrency=1とする。
		def f():
			tro3.v_torch_1d_tensor=t1
			tro3.v_fc=torch.nn.Linear(3,1)
			z1=tro3.v_fc(tro3.v_torch_1d_tensor)
			tro3.torch_no_grad=torch.no_grad()
			#with torch.no_grad(): #torch.no_grad()はスレッドローカルなコンテキストマネージャーである。
			with tro3.torch_no_grad: #そのため、Actor上での実行かつtensorと同じスレッドで実行されなければならない。
					z2=tro3.v_fc(tro3.v_torch_1d_tensor)
			y1=z1+z2
			y1.backward()
			g1=[np.array(p.grad) for p in tro3.v_fc.parameters()]
			tro3.v_fc.zero_grad()
			z1=tro3.v_fc(tro3.v_torch_1d_tensor)
			z1.backward()
			g2=[np.array(p.grad) for p in tro3.v_fc.parameters()]
			tro3.v_fc.zero_grad()
			z1=tro3.v_fc(tro3.v_torch_1d_tensor)
			z2=tro3.v_fc(tro3.v_torch_1d_tensor)
			y1=z1+z2
			y1.backward()
			g3=[np.array(p.grad) for p in tro3.v_fc.parameters()]
			result = np.array([(gg1==gg2).all() and (gg1*2==gg3).all() for gg1,gg2,gg3 in zip(g1,g2,g3)]).all()
			return result
		checkGenericFunc("remote nn.module with torch.no_grad()",f)
	
		if verbose:
			print("========== Additional test for PyTorch succeeded. ==========")
