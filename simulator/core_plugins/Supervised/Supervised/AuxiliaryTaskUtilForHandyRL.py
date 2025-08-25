# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""HandyRLで補助タスク(教師あり/教師なし学習)を行うための基底クラス群
"""
import sys
import collections
import copy
import gymnasium as gym
import random
import bz2
import cloudpickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ASRCAISim1.plugins.HandyRLUtility.model import ModelBase
from ASRCAISim1.plugins.HandyRLUtility.util import map_r, bimap_r, trimap_r, rotate
from ASRCAISim1.plugins.HandyRLUtility.RecurrentBlock import RecurrentBlock

def to_torch(x):
    return map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous() if x is not None else None)

def to_gpu(data):
    return map_r(data, lambda x: x.cuda() if x is not None else None)

def replace_none(a, b):
    return a if a is not None else b

class AuxiliaryTaskManager:
    def __init__(self,args,model):
        self.args=args
        self.model=model
        networks = model.auxiliary_branches if hasattr(model,'auxiliary_branches') else {}
        hiddens = model.init_auxiliary_hidden()
        custom_classes=self.args['custom_classes']
        self.tasks={
            name: custom_classes.get(v.get('class',''),AuxiliaryTaskBase)(name,v,networks[name],hiddens[name])
            for name,v in args['auxiliary_tasks'].items()
        }
        self.policy_to_train = args['policy_to_train']
        #bufferの設定
        self.episodes=[]
        self.capacity=1048576
        self._ep_idx_ofs = 0
        self._next_idx = 0
        self._data_count = 0
        self._total_data_count = 0
        self.ep_indices = [(0,0) for _ in range(self.capacity)]
        self.flat_idx_from_ep = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
        self.inputs_zeros={}
        self.truth_zeros={}
        obs_zeros = map_r(self.args['observation_space'].sample(), lambda x: np.zeros_like(x))
        for name,task in self.tasks.items():
            if(name+'_input' in obs_zeros):
                self.inputs_zeros[name]=obs_zeros[name+'_input']
            if('original' in obs_zeros):
                obs_zeros=obs_zeros['original']
            else:
                aux_keys=[k for k in obs_zeros if k.endswith('_input')]
                obs_zeros={k:v for k,v in obs_zeros.items() if k not in aux_keys}
            self.inputs_zeros['original']=obs_zeros
            if(name in self.args['auxiliary_task_truth_space']):
                self.truth_zeros[name]=map_r(self.args['auxiliary_task_truth_space'][name].sample(),lambda x:np.zeros_like(x))
    def extract_data(self,ep):
        moments = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        players = [idx for idx, policy in enumerate(ep['policy_map'])
            if policy in [self.policy_to_train] + ['Imitator']
        ]
        if len(players) ==0:
            #学習対象のpolicyが含まれない
            return None

        #observationからのinputsの抽出
        obs_zeros = map_r(self.args['observation_space'].sample(), lambda x: np.zeros_like(x))
        obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
        obs = rotate(rotate(obs))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))
        mask = np.array([[[m['observation'][player] is not None] for player in players] for m in moments], dtype=np.float32)
        assert(isinstance(obs,dict))
        inputs={}
        for name,task in self.tasks.items():
            if(name+'_input' in obs):
                inputs[name]=obs[name+'_input']
            if('original' in obs):
                obs=obs['original']
            else:
                aux_keys=[k for k in obs if k.endswith('_input')]
                obs={k:v for k,v in obs.items() if k not in aux_keys}
            inputs['original']=obs
        #truthの抽出
        truths={}
        for name,task in self.tasks.items():
            if(name in self.truth_zeros):
                truth = [[replace_none(t.get(player,None), self.truth_zeros[name]) for player in players] for t in ep['auxiliary_task_truth'][name]]
                truth = rotate(rotate(truth))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
                truth = bimap_r(self.truth_zeros[name], truth, lambda _, t: np.array(t))
                truths[name]=truth
        return {
            'args': ep['args'], 'outcome': ep['outcome'],
            'steps': ep['steps'],
            'policy_map': ep['policy_map'],

            'inputs':inputs,
            'truths':truths,
            'mask':mask,
        }
    def add_episodes(self,episodes):
        ep_cnt = 0
        for ep in episodes:
            extracted=self.extract_data(ep)
            if(extracted is None):
                continue
            self.episodes.append(extracted)
            num_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])
            for train_st in range(num_candidates):
                ep_idx = self._ep_idx_ofs + len(self.episodes) - 1
                self.ep_indices[self._next_idx]=(ep_idx, train_st)
                self.flat_idx_from_ep[ep_idx][train_st] = self._next_idx
                self._next_idx += 1
                self._data_count = min(self._data_count + 1, self.capacity)
                self._total_data_count += 1
                if self._next_idx >= self.capacity:
                    self._next_idx = 0
            ep_cnt += 1
        while self.ep_indices[self._next_idx][0] > self._ep_idx_ofs:
            self.flat_idx_from_ep.pop(self._ep_idx_ofs)
            self._ep_idx_ofs += 1
            self.episodes.popleft()
    def get_episode(self,ep_idx):
        return self.episodes[ep_idx-self._ep_idx_ofs]
    def get_idx_candidates(self):
        return self.flat_idx_from_ep
    def __getitem__(self,idx):
        ep_idx, train_st = self.ep_indices[idx]
        ep = self.get_episode(ep_idx)
        st = max(0, train_st - self.args['burn_in_steps'])
        ed = min(train_st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        return {
            'args': ep['args'], 'outcome': ep['outcome'],
            'inputs':map_r(ep['inputs'],lambda x:x[st:ed]),
            'truths':map_r(ep['truths'],lambda x:x[st:ed]),
            'mask':ep['mask'][st:ed],
            'ep_idx': ep_idx, 'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
            'policy_map': ep['policy_map'],
        }
    def make_batch(self,task_name,episodes):
        inputs_tot=[]
        truth_tot=[]
        mask_tot=[]
        for ep in episodes:
            key = task_name if task_name in ep['inputs'] else 'original'
            inputs = ep['inputs'][key][:ep['end'] - ep['start']]
            if task_name in self.truth_zeros:
                truth = ep['truths'][task_name][:ep['end'] - ep['start']]
            mask = ep['mask'][:ep['end'] - ep['start']]

            # pad each array if step length is short
            batch_steps = self.args['burn_in_steps'] + self.args['forward_steps']
            if len(mask) < batch_steps:
                pad_len_b = self.args['burn_in_steps'] - (ep['train_start'] - ep['start'])
                pad_len_a = batch_steps - len(mask) - pad_len_b
                inputs = map_r(inputs, lambda i: np.pad(i, [(pad_len_b, pad_len_a)] + [(0, 0)] * (len(i.shape) - 1), 'constant', constant_values=0))
                if task_name in self.truth_zeros:
                    truth = map_r(truth, lambda t: np.pad(t, [(pad_len_b, pad_len_a)] + [(0, 0)] * (len(t.shape) - 1), 'constant', constant_values=0))
                mask = np.pad(mask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            inputs_tot.append(inputs)
            if task_name in self.truth_zeros:
                truth_tot.append(truth)
            mask_tot.append(mask)
        inputs = to_torch(bimap_r(self.inputs_zeros[key],rotate(inputs_tot), lambda _, i: np.array(i)))
        if task_name in self.truth_zeros:
            truth = to_torch(bimap_r(self.truth_zeros[key],rotate(truth_tot), lambda _, t: np.array(t)))
        mask = to_torch(np.array(mask_tot))
        return {
            'inputs':inputs,
            'truth':truth,
            'mask':mask
        }
    def forward(self,model,batch):
        inputs = batch['inputs']
        batch_shape = batch['mask'].size()[:3] # (B, T, P or 1)
        hidden = batch.get('hidden_in',None)
        if hidden is None:
            # feed-forward neural network
            inputs = map_r(inputs, lambda i: i.flatten(0, 2)) # (..., B * T * P or 1, ...)
            outputs = model(inputs, None)
            outputs = map_r(outputs, lambda o: o.unflatten(0, batch_shape))  # (..., B, T, P or 1, ...)
        else:
            # sequential computation with RNN
            # RNN block is assumed to be an instance of ASRCAISim1.plugins.HandyRLUtility.RecurrentBlock.RecurrentBlock
            outputs=[]
            seq_len = batch_shape[1]
            hidden = map_r(hidden, lambda h: h.unsqueeze(0).unsqueeze(0).expand((batch_shape[0],batch_shape[2])+h.shape)) # (..., B, P or 1, ...)
            inputs = map_r(inputs, lambda i: i.transpose(1, 2))  # (..., B , P or 1 , T, ...)
            mask_ = map_r(batch['mask'], lambda m: m.transpose(1, 2))  # (..., B , P or 1 , T, ...)
            mask = map_r(hidden, lambda h: mask_.view([*h.size()[:2], seq_len, *([1] * (h.dim() - 2))]).flatten(0,1))
            hidden_ = map_r(hidden, lambda h: h.flatten(0, 1))  # (..., B * P or 1, ...)
            if self.args['burn_in_steps'] >0 :
                model.eval()
                with torch.no_grad():
                    outputs_, hidden_ = model(map_r(inputs, lambda i:i[:, :, :self.args['burn_in_steps'], ...].flatten(0, 2)), hidden_, self.args['burn_in_steps'],mask=map_r(mask,lambda m: m[:,:self.args['burn_in_steps'],...]))
                outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2], self.args['burn_in_steps'])).transpose(1, 2))
                outputs.append(outputs_)
            if not model.training:
                model.train()
            outputs_, hidden_ = model(map_r(inputs, lambda i:i[:, :, self.args['burn_in_steps']:, ...].flatten(0, 2)), hidden_,seq_len-self.args['burn_in_steps'],mask=map_r(mask,lambda m: m[:,self.args['burn_in_steps']:,...]))
            outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2], seq_len-self.args['burn_in_steps'])).transpose(1, 2))
            outputs.append(outputs_)
            outputs = torch.cat(outputs, dim=1).mul(batch['mask'])
        return outputs
    def train(self):
        losses={}
        metrics={}
        for name,task in self.tasks.items():
            losses[name],m=task.train(self)
            if(len(m)>0):
                metrics[name]=m
        return losses,metrics

class AuxiliaryTaskBase:
    def __init__(self, task_name, task_config, network, hidden):
        self.task_config = copy.deepcopy(task_config)
        self.task_name=task_name
        self.network = network
        self.initial_hidden = hidden
        self.batch_size = task_config.get('batch_size',128)
        self.params = list(self.network.parameters())
        lr=task_config.get('lr',1e-4)
        weight_decay=task_config.get('weight_decay',1e-5)
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=weight_decay) if len(self.params) > 0 else None
    def select_episode_indices(self, manager):
        candidates=manager.get_idx_candidates()
        def selector():
            ep_idx=random.randrange(len(candidates))
            train_st=random.randrange(len(candidates[ep_idx]))
            return candidates[ep_idx][train_st]
        return [selector() for i in range(self.batch_size)]
    def loss_and_metrics(self, output, batch, manager):
        #バッチのlossとmetrics(dict)を返す
        raise NotImplementedError
    def train(self, manager):
        episodes=[manager[idx] for idx in self.select_episode_indices(manager)]
        batch=manager.make_batch(self.task_name,episodes)
        batch['hidden_in']=to_torch(self.initial_hidden)
        if(torch.cuda.device_count()>0):
            batch=to_gpu(batch)
        outputs=manager.forward(self.network,batch)
        loss,metrics=self.loss_and_metrics(outputs,batch,manager)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 4.0)
        self.optimizer.step()
        return loss,metrics

class AuxiliaryTaskNetworkBase(nn.Module):
    def __init__(self, input_space, truth_space, model_classes, model_config):
        super().__init__()
    @property
    def output_shape(self):
        raise NotImplementedError
    def forward(self, inputs, states=None, seq_len=None, mask=None):
        raise NotImplementedError
    def init_hidden(self, batch_size=None):
        raise NotImplementedError
