# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""ray RLlibで学習するためのtorch.nn.Moduleのサンプル。
R7ContestAgentSample01のobservation,actionに対応したものとなっている。

* friend_enemy_relativeは使用していない。
  このサンプルでは各Assetのembeddingを作成してpermutation equivariance/invarianceな操作を行う。
  要素数Nfのシーケンスと要素数Neのシーケンスに対して[Nf,Ne,D]のshapeを持った付加情報を加えるためには、
  様々な方式が考えられるが、シンプルに実装できるものは乏しい。
  例えば、attention weightにバイアスを加える方法もあるが、この機能はPyTorchのMultiheadAttentionには存在しない。

"""
import copy
import logging
import numpy as np
from typing import Dict, Optional, Union
import gymnasium as gym

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.recurrent_net import LSTMWrapper
from ray.rllib.models.torch.torch_modelv2 import ModelV2,TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from .GenericTorchModelUtil import (
    GenericLayers,
    SetTransformer,
    SetTransformerWithQuery,
    TransformerEncoder,
    TransformerDecoder
)

class R7ContestTorchNNSampleForRay(TorchModelV2, nn.Module):
    """Wrapperでなく、これ自身が中核
    """
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str,
                 **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.custom_model_config=copy.deepcopy(kwargs)
        orig_space=getattr(obs_space, "original_space", obs_space)

        if not isinstance(orig_space,gym.spaces.Dict):
            raise ValueError("Invalid observation space. orig_space={}".format(orig_space))
        if not isinstance(action_space,gym.spaces.Tuple) or not isinstance(action_space[0],gym.spaces.Dict):
            raise ValueError("Invalid action space. action_space={}".format(action_space))

        self.core_dim=0

        if 'common' in orig_space.spaces:
            #common branch
            if 'common' in self.custom_model_config:
                cfg=self.custom_model_config['common']
                cfg['input_shape']=orig_space['common'].shape
                self.common=GenericLayers(cfg)
                self.core_dim+=self.common.output_shape[-1]

        if 'image' in orig_space.spaces:
            #image branch
            if 'image' in self.custom_model_config:
                cfg=self.custom_model_config['image']
                cfg['input_shape']=orig_space['image'].shape
                self.image=GenericLayers(cfg)
                self.core_dim+=self.image.output_shape[-1]

        self.entity_embeds={}
        for key in ['parent','friend','enemy','friend_missile','enemy_missile']:
            if key in orig_space.spaces:
                #entity embedding
                if key in self.custom_model_config:
                    cfg=self.custom_model_config[key]
                    cfg['input_shape']=orig_space[key].shape
                    embed=GenericLayers(cfg)
                    setattr(self,key+'_embed',embed)
                    self.entity_embeds[key]=embed

        # friend_enemy_relative (shape=[Np+Nf,Ne,Dfe])はこのサンプルでは使用しない。
        # 例えばAttention weightに加算する等の用途が考えられる。

        if len(self.entity_embeds)>0:
            # embed_dimの整合確認
            self.embed_dim=next(iter(self.entity_embeds.values())).output_shape[-1]
            for embed in self.entity_embeds.values():
                assert self.embed_dim==embed.output_shape[-1]

            #dummy entity (「射撃しない」等を表現するための空のエンティティ)
            self.dummy_entity = torch.nn.Parameter(torch.empty((1,1,self.embed_dim), dtype=torch.float32))
            bound = 1 / np.sqrt(self.embed_dim)
            torch.nn.init.uniform_(self.dummy_entity,-bound,bound)

            #permutation equivariant transformation
            if 'entity_equivariant' in self.custom_model_config:
                cfg=self.custom_model_config['entity_equivariant']
                cfg['input_shape']=[1,self.embed_dim]
                if cfg['type']=='SetTransformer':
                    self.entity_equivariant=SetTransformer(cfg)
                elif cfg['type']=='TransformerEncoder':
                    self.entity_equivariant=TransformerEncoder(cfg)
                else:
                    raise ValueError('type of entity_equivariant needs to be SetTransformer or TransformerEncoder.')

            #permutation invariant transformation
            cfg=self.custom_model_config['entity_invariant']
            cfg['input_shape']=[1,self.embed_dim]
            if cfg['type']=='SetTransformer':
                self.entity_invariant=SetTransformer(cfg)
                assert self.entity_invariant.hasPMA and self.entity_invariant.num_seeds==1
            elif cfg['type']=='SetTransformerWithQuery':
                self.entity_invariant=SetTransformerWithQuery(cfg)
                self.entity_invariant_query = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))
                nn.init.xavier_uniform_(self.entity_invariant_query)
            elif cfg['type']=='TransformerDecoder':
                self.entity_invariant=TransformerDecoder(cfg)
                self.entity_invariant_query = nn.Parameter(torch.Tensor(1, 1, self.embed_dim))
                nn.init.xavier_uniform_(self.entity_invariant_query)
            else:
                raise ValueError('type of entity_invariant needs to be SetTransformer, SetTransformerWithQuery or TransformerDecoder.')

            self.core_dim+=self.entity_invariant.output_shape[-1]

        #merge branch
        self.use_merger = 'merge' in self.custom_model_config
        if self.use_merger:
            cfg=self.custom_model_config['merge']
            cfg['input_shape']=[self.core_dim]
            self.merge=GenericLayers(cfg)
            self.merged_dim=self.merge.output_shape[-1]
        else:
            self.merged_dim=self.core_dim
        
        #recurrent block
        self.use_lstm = self.custom_model_config.get("use_lstm",False)
        if(self.use_lstm):
            self.time_major = model_config.get("_time_major", False)
            in_space =gym.spaces.Box(
                float("-inf"),
                float("inf"),
                shape=(self.merged_dim, ),
                dtype=np.float32)
            self.lstm_cell_size = self.custom_model_config.get("lstm_cell_size",64)
            class PureLSTM(LSTMWrapper):
                pass
            def dummy_forward(
                self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType,
            ):
                return input_dict["obs_flat"], state
            PureLSTM._wrapped_forward = dummy_forward
            self.core_recurrent_block=PureLSTM(
                in_space,
                action_space,
                self.merged_dim,
                {
                    "_time_major":self.time_major,
                    "lstm_cell_size":self.lstm_cell_size,
                    "lstm_use_prev_action":False,
                    "lstm_use_prev_reward":False
                },
                "core_recurrent_block"
            )

        #apply merged (recurrent) info to entities
        #この例ではrecurrentの出力をKV(memory)、entityのembeddingをQ(tgt)としてTransformerDecoderに流す
        #また、この例ではこれより後でparentとenemyしか使用しないため、いずれかが存在する場合のみ処理する
        if 'parent' in self.entity_embeds or 'enemy' in self.entity_embeds:
            if 'apply_merged_to_entity' in self.custom_model_config:
                if(self.embed_dim!=self.merged_dim):
                    cfg={
                        'input_shape':[self.merged_dim],
                        'layers':[
                            ['Linear',{'out_features':self.embed_dim}]
                        ]
                    }
                    self.apply_merged_to_entity_pre=GenericLayers(cfg)

                cfg=self.custom_model_config['apply_merged_to_entity']
                cfg['input_shape']=[1,self.embed_dim]
                if cfg['type']=='SetTransformerWithQuery':
                    self.apply_merged_to_entity=SetTransformerWithQuery(cfg)
                elif cfg['type']=='TransformerDecoder':
                    self.apply_merged_to_entity=TransformerDecoder(cfg)
                else:
                    raise ValueError('type of apply_merged_to_entity needs to be SetTransformerWithQuery or TransformerDecoder.')

        #critic branch
        self.critic_heads={}
        for key in self.custom_model_config['critic']:
            cfg=self.custom_model_config['critic'][key]
            cfg['input_shape']=[self.merged_dim]
            cfg['layers'].append(['Linear',{'out_features':1}])
            head=GenericLayers(cfg)
            setattr(self,'critic_'+key+'_head',head)
            self.critic_heads[key]=head

        #actor branch
        if 'parent' in self.entity_embeds:
            #このサンプルでは、parentのembeddingを作る場合、全parentに対して同一の重みで行動を計算する。
            self.action_heads={}
            for key in action_space[0].spaces:
                if key=='target' and 'enemy' in self.entity_embeds:
                    # enemyのembeddingを作る場合の目標選択
                    # (1) 各parentを長さ1のKVとして、dummy+enemyをQとしたTransformerDecoderを流す。
                    #   [B*N_parent,1,self.embed_dim], [B*N_parent,1+N_enemy,self.embed_dim] -> [B*N_parent,1+N_enemy,self.embed_dim]
                    # (2) Linearで末尾の次元を1にして、squeeze(-1)とunflatten(0,[B,N_parent])によって、
                    #   [B,N_parent,1+N_enemy]の出力を得る。
                    cfg=self.custom_model_config['actor'][key]
                    cfg['input_shape']=[1,self.embed_dim]
                    cfg['layers'].append(['Linear',{'out_features':1}])
                    if cfg['type']=='SetTransformerWithQuery':
                        head=SetTransformerWithQuery(cfg)
                    elif cfg['type']=='TransformerDecoder':
                        head=TransformerDecoder(cfg)
                    else:
                        raise ValueError('type of actor["'+key+'"] needs to be SetTransformerWithQuery or TransformerDecoder.')
                    setattr(self,'actor_'+key+'_head',head)
                    self.action_heads[key]=head
                else:
                    # それ以外はshape=[B,N_parent,self.embed_dim]のembed_dimの次元をデータ次元としてGenericLayersを用いる。
                    # 出力のshapeは[B,N_parent,sub_action_dim]となる。
                    cfg=self.custom_model_config['actor'][key]
                    cfg['input_shape']=[self.embed_dim]
                    _, sub_action_dim = ModelCatalog.get_action_dist(action_space[0][key], model_config, framework='torch')
                    cfg['layers'].append(['Linear',{'out_features':sub_action_dim}])
                    head=GenericLayers(cfg)
                    setattr(self,'actor_'+key+'_head',head)
                    self.action_heads[key]=head
        else:
            #もしparentのembeddingを作らない場合はmergedから単純な全結合で、actionに必要なパラメータ数が出力されるように計算する。
            cfg=self.custom_model_config['actor']
            cfg['input_shape']=[self.merged_dim]
            cfg['layers'].append(['Linear',{'out_features':self.num_outputs}])
            self.action_head=GenericLayers(cfg)

        if(self.use_lstm):
            self.view_requirements = self.core_recurrent_block.view_requirements
            self.view_requirements["obs"].space = self.obs_space

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs=input_dict[SampleBatch.OBS]

        core=[]
        B=None #バッチサイズ

        if 'common' in obs and hasattr(self,'common'):
            B=obs['common'].shape[0]
            core.append(self.common(obs['common'])) #[B,Dc]→[B,D_common]

        if 'image' in obs and hasattr(self,'image'):
            B=obs['image'].shape[0]
            core.append(self.image(obs['image'])) #[B,Ch,Lon,Lat]→[B,D_image]

        embedded={} #{key:[B,N_key,self.embed_dim]}
        entity_mask={} #{key:[B,N_key,1]}
        if len(self.entity_embeds)>0:
            for key in ['parent','friend','enemy','friend_missile','enemy_missile']:
                if key in obs and key in self.entity_embeds:
                    B=obs[key].shape[0]
                    embedded[key]=self.entity_embeds[key](obs[key])

                    if 'observation_mask' in obs:
                        if key in obs['observation_mask']:
                            #PytorchではTrueのときに無効化される。obsは有効なものを1としているので反転する。
                            entity_mask[key]=obs['observation_mask'][key]!=1 #[B,N_key]
                        else:
                            #observation maskが与えられていなければ全て有効(False)とする
                            N=embedded[key].shape[-2]
                            entity_mask[key]=torch.full((B,N),False,device=self.dummy_entity.device)
            embedded['dummy']=self.dummy_entity.expand(B,1,-1)
            entity_mask['dummy']=torch.full((B,1),False,device=self.dummy_entity.device)

            cat_embedded=torch.cat(list(embedded.values()),dim=-2) #[B,N,self.embed_dim]
            cat_entity_mask=torch.cat(list(entity_mask.values()),dim=-1) #[B,N]
            if 'entity_equivariant' in obs and hasattr(self,'entity_equivariant'):
                cat_embedded=self.entity_equivariant(
                    src=cat_embedded,
                    src_key_padding_mask=cat_entity_mask
                ) #[B,N,self.embed_dim]

            if isinstance(self.entity_invariant,SetTransformer):
                core.append(torch.reshape(self.entity_invariant(
                    src=cat_embedded,
                    src_key_padding_mask=cat_entity_mask
                ),[B,self.embed_dim])) #[B,self.embed_dim]
            else:
                # Queryが必要
                query = self.entity_invariant_query.repeat(B, 1, 1)
                core.append(torch.reshape(self.entity_invariant(
                    tgt=query,
                    memory=cat_embedded,
                    memory_key_padding_mask=cat_entity_mask
                ),[B,self.embed_dim])) #[B,self.embed_dim]

        #merge
        merged=torch.cat(core,dim=-1)+0 #[B,self.core_dim]
        if self.use_merger:
            merged=self.merge(merged)

        #recurrent
        if(self.use_lstm):
            input_dict["obs_flat"]=input_dict["obs"]=merged
            merged,state_out=self.core_recurrent_block(input_dict,state,seq_lens)#[B×D_core]
        else:
            state_out=state

        #apply merged (recurrent) info to entities
        if 'parent' in embedded or 'enemy' in embedded:
            if hasattr(self,'apply_merged_to_entity'):
                if hasattr(self,'apply_merged_to_entity_pre'):
                    kv=torch.reshape(self.apply_merged_to_entity_pre(merged),[B,1,self.embed_dim])
                else:
                    kv=torch.reshape(merged,[B,1,self.embed_dim])
                for key in ['parent','enemy']:
                    if key in embedded:
                        embedded[key]=self.apply_merged_to_entity(
                            tgt=embedded[key],
                            memory=kv
                        ) #[B,N_key,self.embed_dim]
        
        #actor
        if 'parent' in self.entity_embeds:
            #parentのembeddingを作る場合
            parent_action_list=[]
            for key in self.action_space[0].spaces:
                if key=='target' and 'enemy' in self.entity_embeds:
                    # (1) 各parentを長さ1のKVとして、dummy+enemyをQとしたTransformerDecoderを流す。
                    #   [B*N_parent,1,self.embed_dim], [B*N_parent,1+N_enemy,self.embed_dim] -> [B*N_parent,1+N_enemy,self.embed_dim]
                    # (2) Linearで末尾の次元を1にして、squeeze(-1)とunflatten(0,[B,N_parent])によって、
                    #   [B,N_parent,1+N_enemy]の出力を得る。
                    num_parents=embedded['parent'].shape[-2]
                    tgt=torch.cat([embedded['dummy'],embedded['enemy']],dim=-2)
                    tgt=tgt.unsqueeze(1).repeat(1, num_parents, 1, 1).flatten(0,1)
                    memory=embedded['parent'].flatten(0,1).unsqueeze(1)

                    logits=self.action_heads[key](
                        tgt=tgt,
                        memory=memory
                    ).squeeze(-1).unflatten(0,[B,num_parents]) #[B,N_parent,1+N_enemy]
                    
                    # action maskを適用し、無効な対象が選ばれないようにする(-infにする)
                    # サンプラーが-infになっている選択肢を選ばない仕様になっていることを確認すること！
                    if self.custom_model_config['actor']['target'].get('apply_mask',True):
                        target_mask=torch.cat([entity_mask['dummy'],entity_mask['enemy']],dim=-1) #[B,1+N_enemy]
                        target_mask=target_mask.unsqueeze(1).repeat(1, num_parents, 1) #[B,N_parent,1+N_enemy]
                        logits=torch.where(target_mask,-torch.inf,logits)

                    parent_action_list.append(logits)

                else:
                    logits=self.action_heads[key](embedded['parent'])

                    # 要すればaction maskを適用し、無効な対象が選ばれないようにする(-infにする)
                    # サンプラーが-infになっている選択肢を選ばない仕様になっていることを確認すること！
                    if (
                        'action_mask' in obs and key in obs['action_mask'][0]
                        and self.custom_model_config['actor'][key].get('apply_mask',True)
                    ):
                        mask=torch.stack([m[key] for m in obs['action_mask']],dim=1)

                        # all zeroのdummy dataが流れてきたときは全て1(有効にする)
                        mask=torch.where(torch.sum(mask,dim=-1,keepdim=True)==0,1,mask)

                        logits=torch.where(mask!=1,-torch.inf,logits)
                    parent_action_list.append(logits) #[B,N_parent,sub_action_dim]
            parent_action=torch.cat(parent_action_list,dim=-1) #[B,N_parent,parent_action_dim]
            
            #parentのentity_maskを反映し、生存していないparentのactionから勾配が流れないようにする
            parent_action=torch.where(
                entity_mask['parent'].unsqueeze(-1),
                parent_action.detach(),
                parent_action
            )
            
            #元のaction spaceの形(Tuple[Dict])に合わせて並び替える
            action_logits=torch.cat(
                [parent_action[:,i,...] for i in range(len(self.action_space))],
                dim=-1) #[B,total_action_dim]
        else:
            #parentのembeddingを作らない場合
            action_logits=self.action_head(merged)

        #critic branch
        self._value_base_features=merged+0

        return action_logits,state_out

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        if(self.use_lstm):
            return self.core_recurrent_block.get_initial_state()
        else:
            return []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._value_base_features is not None, "must call forward() first"
        x=self._value_base_features
        value=self.critic_heads['value'](x)
        return value.squeeze(1)

ModelCatalog.register_custom_model("R7ContestTorchNNSampleForRay",R7ContestTorchNNSampleForRay)
