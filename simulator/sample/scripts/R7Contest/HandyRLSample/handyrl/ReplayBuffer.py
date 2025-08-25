# ===== Original Version (ReplayBuffer's default routine as a part of trainer.py of HandyRL) =====
# Copyright (c) 2020 DeNA Co., Ltd.
#
# ===== Modified Version (Separated ReplayBuffer interface and PrioritizedReplayBuffer) =====
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

from math import ceil, log
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque, defaultdict
import warnings
import psutil
import bz2
import cloudpickle

from ray.rllib.execution.segment_tree import SumSegmentTree, MinSegmentTree, SegmentTree

class ReplayBuffer:
    """
    HandyRL改において、優先度付き経験再生を含む様々なバッチ生成を可能とするための、リプレイバッファを表現する基底クラス。
    """
    def __init__(self, args):
        self.args = args
        self.episodes = deque()
        self._data_count = 0
        self._total_data_count = 0
    def get_current_data_count(self):
        #現在バッファに格納されているデータ点の数を返す。
        return self._data_count
    def get_total_data_count(self):
        #これまでに追加されたデータ点の総数を返す。
        return self._total_data_count
    def is_ready(self):
        #リプレイバッファが読み出し可能かどうかを返す。
        raise NotImplementedError()
    def get_episode(self,ep_idx):
        #指定されたインデックスに対応するエピソードを返す。
        raise NotImplementedError()
    def add_episodes(self,episodes):
        #エピソードのリストを与えてバッファに追加する。
        for ep in episodes:
            if self.args.get('replay_buffer_config',{}).get('independent_replay_for_each_player',False):
                # playerごとに独立したデータとして追加する場合
                assert(self.args.get('replay_buffer_config',{}).get('discard_untrainable_players',False))
                moments_all = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
                policy_map=ep['policy_map']
                for player in moments_all[0]['observation']:
                    modified = {k: v for k, v in ep.items() if not k in ['moment', 'policy_map']}
                    moments = [{k: {player: v[player]} if k != 'turn' else v for k, v in m.items()} for m in moments_all]
                    if self.args.get('replay_buffer_config',{}).get('ignore_noaction_timesteps',False):
                        # playerが行動していない時刻のデータを消して詰める場合
                        moments = [{k: v for k, v in m.items()} for m in moments_all if player in m['turn']]

                    if len(moments)>0:
                        modified['steps']=len(moments)
                        modified['moment'] = [
                            bz2.compress(cloudpickle.dumps(moments[i:i+self.args['compress_steps']]))
                            for i in range(0, len(moments), self.args['compress_steps'])
                        ]
                        modified['policy_map'] = [policy if idx==player else "" for idx, policy in enumerate(policy_map)]
                        self.add_episodes_sub(modified, self.calcPriority(modified))
            else:
                # 全player分を一つのデータとして追加する場合
                moments_all = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
                modified = {k: v for k, v in ep.items() if not k in ['moment']}
                if self.args.get('replay_buffer_config',{}).get('ignore_noaction_timesteps',False):
                    # 誰も行動していない時刻のデータを消して詰める場合
                    moments = [{k: v for k, v in m.items()} for m in moments_all if len(m['turn'])>0]
                if len(moments)>0:
                    modified['steps']=len(moments)
                    modified['moment'] = [
                        bz2.compress(cloudpickle.dumps(moments[i:i+self.args['compress_steps']]))
                        for i in range(0, len(moments), self.args['compress_steps'])
                    ]
                    self.add_episodes_sub(ep, self.calcPriority(ep))
    def add_episodes_sub(self,ep,priorities = None):
        #エピソードと初期優先度の組を与えてバッファに追加する。
        raise NotImplementedError()
    def select_episode(self):
        #バッファから一つのデータ点を選択し、エピソードを切り出したシーケンスを返す。
        raise NotImplementedError()
    def calcPriority(self,ep):
        #エピソードの初期優先度を必要に応じ計算する。
        if(self.args.get('replay_buffer_config',{}).get('calc_initial_priority',False)):
            ep_idx = 0
            train_st = 0
            st = 0
            ed = ep['steps']
            st_block = st // self.args['compress_steps']
            ed_block = (ed - 1) // self.args['compress_steps'] + 1
            ep_for_dummy_batch = [{
                'args': ep['args'], 'outcome': ep['outcome'],
                'moment': ep['moment'][st_block:ed_block],
                'base': st_block * self.args['compress_steps'],
                'ep_idx': ep_idx, 'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
                'policy_map': ep['policy_map'],
            }]
            ep_num_candidates = [1 + max(0, ep['steps'] - self.args['forward_steps'])]
            target_errors_for_priority = ep.pop('target_errors_for_priority')
            advantages_for_priority = ep.pop('advantages_for_priority')
            prio = self.calcPrioritySub(target_errors_for_priority, advantages_for_priority,ep_num_candidates,self.args)[0]
        else:
            prio = None
        return prio
    def calcPrioritySub(self,target_errors,advantages,ep_num_candidates,args):
        #エピソード全体のシーケンスから計算されたtarget_errorとadvantageを用いて各時刻の初期優先度を計算する。
        #[B,T,P or 1,1]で入ってくる
        raise NotImplementedError()
    def update(self,ep_indices,target_errors,advantages):
        #エピソードから切り出された固定長シーケンスのバッチから計算されたtarget_errorとadvantageを用いて各々の新たな優先度を計算する。
        #[B,T,P or 1,1]で入ってくる
        raise NotImplementedError()
    def get_metrics(self):
        #SummaryWriter等でログとして記録すべき値をdictで返す。
        return {
            'mem_percent': psutil.virtual_memory().percent,
            'current_data_count': self.get_current_data_count(),
            'total_data_count': self.get_total_data_count(),
        }

class DefaultReplayBuffer(ReplayBuffer):
    """
    オリジナルのHandyRLと同じ方法で経験再生を行うクラス。
    """
    def __init__(self, args):
        super().__init__(args)
        if 'maximum_episodes' in self.args:
            self.maximum_episodes = self.args['maximum_episodes']
        elif 'replay_buffer_config' in self.args:
            config = self.args['replay_buffer_config']
            if 'maximum_episodes' in config:
                self.maximum_episodes = config['maximum_episodes']
    def is_ready(self):
        return len(self.episodes) >= self.args['minimum_episodes']
    def get_episode(self,ep_idx):
        return self.episodes[ep_idx]
    def add_episodes_sub(self,ep,priorities = None):
        mem_percent = psutil.virtual_memory().percent

        mem_ok = mem_percent <= 95
        maximum_episodes = self.maximum_episodes if mem_ok else int(len(self.episodes) * 95 / mem_percent)

        if not mem_ok and 'memory_over' not in self.flags:
            warnings.warn("memory usage %.1f%% with buffer size %d" % (mem_percent, len(self.episodes)))
            self.flags.add('memory_over')

        while len(self.episodes) > maximum_episodes:
            self.episodes.popleft()

        self.episodes.append(ep)
        num_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])
        self._data_count += num_candidates
        self._total_data_count += num_candidates
        while len(self.episodes) >= maximum_episodes:
            popped=self.episodes.popleft()
            self._data_count -= 1 + max(0, popped['steps'] - self.args['forward_steps'])
    def select_episode_index(self):
        while True:
            ep_count = min(len(self.episodes), self.maximum_episodes)
            ep_idx = random.randrange(ep_count)
            accept_rate = 1 - (ep_count - 1 - ep_idx) / self.maximum_episodes
            if random.random() < accept_rate:
                break
        ep = self.get_episode(ep_idx)
        turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        train_st = random.randrange(turn_candidates)
        return ep_idx,train_st
    def select_episode(self):
        ep_idx, train_st = self.select_episode_index()
        ep = self.get_episode(ep_idx)
        st = max(0, train_st - self.args['burn_in_steps'])
        ed = min(train_st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        return {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'ep_idx': ep_idx, 'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
            'policy_map': ep['policy_map'],
        }
    def calcPrioritySub(self,target_errors,advantages,ep_num_candidates,args):
        return [
            [
                1.0
                for t in range(steps)
            ] for steps in ep_num_candidates
        ]
    def update(self,ep_indices,target_errors,advantages):
        pass
    def get_metrics(self):
        #SummaryWriter等でログとして記録すべき値をdictで返す。
        ret=super().get_metrics()
        ret['num_episodes']=len(self.episodes)
        return ret

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    優先度付き経験再生を行うリプレイバッファ。ray.rllibのSumTreeを使用している。
    """
    def __init__(self, args):
        super().__init__(args)
        if 'replay_buffer_config' in self.args:
            config = self.args['replay_buffer_config']
        else:
            config = {}
        capacity=config.get('capacity', 1e6)
        self.capacity = int(2 ** ceil(log(capacity, 2))) #SumTreeのcapacityは2のべき乗に固定
        self.default_priority = config.get('default_priority', 1.0)
        self.alpha = config.get('alpha', 1.0)
        self.beta_start = config.get('beta_start', config.get('beta', 0.1))
        self.beta_end = config.get('beta_end', config.get('beta', 1.0))
        self.beta_decay = config.get('beta_decay', self.capacity)
        self.store_interval = config.get('store_interval', 1)
        self.randomize_offset = config.get('randomize_offset', False)
        self.eta = config.get('eta', 0.9)
        self.eps = config.get('eps', 1e-8)
        self._ep_idx_ofs = 0
        self._next_idx = 0
        self._data_count = 0
        self._total_data_count = 0
        self.max_tree = SegmentTree(self.capacity, operation=max)
        self.sum_tree = SumSegmentTree(self.capacity)
        self.min_tree = MinSegmentTree(self.capacity)
        self.ep_indices = [(0,0) for _ in range(self.capacity)]
        self.tree_idx_from_ep = defaultdict(lambda: defaultdict(lambda: None))
    def is_ready(self):
        return len(self.episodes) >= self.args['minimum_episodes'] or self._data_count == self.capacity
    def get_episode(self,ep_idx):
        return self.episodes[ep_idx-self._ep_idx_ofs]
    def add_episodes_sub(self,ep,priorities = None):
        self.episodes.append(ep)
        train_st_offset = random.randint(0, self.store_interval - 1) if self.randomize_offset else 0
        num_candidates = ceil((1 + max(0, ep['steps'] - self.args['forward_steps'] - train_st_offset))/self.store_interval)
        for train_st_idx in range(num_candidates):
            train_st = train_st_idx*self.store_interval + train_st_offset
            if priorities is None:
                weight = self.default_priority
            else:
                weight = priorities[train_st]
            self.max_tree[self._next_idx]=weight ** self.alpha
            self.sum_tree[self._next_idx]=weight ** self.alpha
            self.min_tree[self._next_idx]=weight ** self.alpha
            ep_idx = self._ep_idx_ofs + len(self.episodes) - 1
            self.ep_indices[self._next_idx]=(ep_idx, train_st)
            self.tree_idx_from_ep[ep_idx][train_st] = self._next_idx
            self._next_idx += 1
            self._data_count = min(self._data_count + 1, self.capacity)
            self._total_data_count += 1
            if self._next_idx >= self.capacity:
                self._next_idx = 0
        while self.ep_indices[self._next_idx][0] > self._ep_idx_ofs:
            self.tree_idx_from_ep.pop(self._ep_idx_ofs)
            self._ep_idx_ofs += 1
            self.episodes.popleft()
    def select_episode_index(self):
        mass = random.random() * self.sum_tree.sum(0, self._data_count)
        idx = self.sum_tree.find_prefixsum_idx(mass)
        return self.ep_indices[idx]
    def select_episode(self):
        ep_idx, train_st = self.select_episode_index()
        ep = self.get_episode(ep_idx)
        st = max(0, train_st - self.args['burn_in_steps'])
        ed = min(train_st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        ret = {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'ep_idx': ep_idx, 'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
            'policy_map': ep['policy_map'],
        }
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0,self._total_data_count / self.beta_decay)
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self._data_count)**(-beta)
        ep_idx = ret['ep_idx']
        train_st = ret['train_start']
        idx = self.tree_idx_from_ep[ep_idx][train_st]
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self._data_count)**(-beta)
        ret['importance_weight'] = weight / max_weight
        return ret
    def calcPrioritySub(self,target_errors,advantages,ep_num_candidates,args):
        #エピソード全体のシーケンスを与えて各時刻の初期優先度を計算する。
        #[B,T,P or 1,1]で入ってくる
        with torch.no_grad():
            as_list = [torch.abs(v) for v in (*target_errors.values(),*advantages.values())]
            mean = sum(as_list)/len(as_list)
            #B個の[1,T,P or 1,1]に分割
            deltas = torch.split(mean, 1, dim=0)
            prio = [
                [
                    self.eta*torch.max(d[:,t:t+args['forward_steps'],...]) + (1-self.eta)*torch.mean(d[:,t:t+args['forward_steps'],...]) + self.eps
                    for t in range(steps)
                ] for d,steps in zip(deltas,ep_num_candidates)
            ]
        return prio

    def update(self,ep_indices,target_errors,advantages):
        #エピソードから切り出された固定長シーケンスのバッチを与えて、各々の新たな優先度を計算する。
        #[B,T,P or 1,1]で入ってくる
        with torch.no_grad():
            as_list = [torch.abs(v) for v in (*target_errors.values(),*advantages.values())]
            mean = sum(as_list)/len(as_list)
            #B個の[T,P or 1,1]に分割
            deltas = torch.split(mean, 1, dim=0)
            deltas = [self.eta*torch.max(d) + (1-self.eta)*torch.mean(d) for d in deltas]

            for i in range(len(ep_indices)):
                ep_idx, train_st = ep_indices[i]
                priority=float(deltas[i])+self.eps
                idx = self.tree_idx_from_ep[ep_idx][train_st]
                if(idx is not None):
                    #update前にbufferから消されていた場合はtreeを更新しない
                    self.max_tree[idx] = priority**self.alpha
                    self.sum_tree[idx] = priority**self.alpha
                    self.min_tree[idx] = priority**self.alpha
                    self.default_priority = self.max_tree.reduce()
    def get_metrics(self):
        #SummaryWriter等でログとして記録すべき値をdictで返す。
        ret=super().get_metrics()
        ret['max_priority']=self.max_tree.reduce()
        ret['mean_priority']=self.sum_tree.sum()/self._data_count
        ret['min_priority']=self.min_tree.min()
        return ret
