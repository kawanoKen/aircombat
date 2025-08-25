# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# batcher

import time
import bz2
import cloudpickle

import numpy as np
import torch
import ray
from ray.util.queue import Queue
from .util import map_r, bimap_r, trimap_r, rotate
from .model import to_torch

def replace_none(a, b):
    return a if a is not None else b

def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (B, T, P, ...) .
        (B is batch size, T is time length, P is player count)
    """

    obss, datum = [], []
    hidden_ins, probs, acts, lacts = [], [], [], []

    policy_to_train = args['policy_to_train']
    ep_indices = []

    for ep in episodes:
        moments_ = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        moments = moments_[ep['start'] - ep['base']:ep['end'] - ep['base']]
        players = [idx for idx, policy in enumerate(ep['policy_map'])
            if policy in [policy_to_train] + ['Imitator']
        ]
        if len(players) ==0:
            continue

        # template for padding
        obs_zeros = map_r(args['observation_space'].sample(), lambda x: np.zeros_like(x))
        action_zeros = args['action_dist_class'].getDefaultAction(args['action_space'])
        prob_ones = map_r(action_zeros, lambda x: np.zeros_like(x))
        legal_actions_zeros = args['action_dist_class'].getDefaultLegalActions(args['action_space'])

        # data that is changed by training configuration (ASRC: Disabled turn-based configuration.)
        obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
        prob = [[replace_none(m['selected_prob'][player], prob_ones) for player in players] for m in moments]
        act = [[replace_none(m['action'][player], action_zeros) for player in players] for m in moments]
        lact = [[replace_none(m['legal_actions'][player], legal_actions_zeros) for player in players] for m in moments]
        initial_hidden = args['initial_hidden']
        record_hidden_in = args.get('record_hidden_in',True) and initial_hidden is not None
        if(record_hidden_in):
            hidden_in = [replace_none(moments[0]['hidden_in'][player], initial_hidden) for player in players]

        # reshape observation etc.
        obs = rotate(rotate(obs))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))
        prob = rotate(rotate(prob))
        prob = bimap_r(prob_ones, prob, lambda _, p: np.array(p))
        act = rotate(rotate(act))
        act = bimap_r(action_zeros, act, lambda _, a: np.array(a))
        lact = rotate(rotate(lact))
        lact = bimap_r(legal_actions_zeros, lact, lambda _, a: np.array(a))
        if(record_hidden_in):
            hidden_in = rotate(hidden_in) # (P, ..., ...) -> (..., P, ...)
            hidden_in = bimap_r(initial_hidden, hidden_in, lambda _, h: np.array(h))

        # datum that is not changed by training configuration
        v = np.array([[replace_none(m['value'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        rew = np.array([[replace_none(m['reward'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        ret = np.array([[replace_none(m['return'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        oc = np.array([ep['outcome'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)

        emask = np.ones((len(moments), 1, 1), dtype=np.float32)  # episode mask
        tmask = np.array([[[m['selected_prob'][player] is not None] for player in players] for m in moments], dtype=np.float32)
        omask = np.array([[[m['observation'][player] is not None] for player in players] for m in moments], dtype=np.float32)

        # Imitation flag
        imi = np.array([[[1.0 if ep['policy_map'][player] == 'Imitator' else 0.0] for player in players] for m in moments], dtype=np.float32)

        progress = np.arange(ep['start'], ep['end'], dtype=np.float32)[..., np.newaxis] / ep['total']

        # pad each array if step length is short
        batch_steps = args['burn_in_steps'] + args['forward_steps']
        if len(tmask) < batch_steps:
            pad_len_b = args['burn_in_steps'] - (ep['train_start'] - ep['start'])
            pad_len_a = batch_steps - len(tmask) - pad_len_b
            obs = map_r(obs, lambda o: np.pad(o, [(pad_len_b, pad_len_a)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
            prob = bimap_r(prob_ones, prob, lambda p_o, p: np.concatenate([np.tile(p_o, [pad_len_b, p.shape[1]]+[1]*(len(p.shape) - 2)), p, np.tile(p_o, [pad_len_a, p.shape[1]]+[1]*(len(p.shape) - 2))]))
            v = np.concatenate([np.pad(v, [(pad_len_b, 0), (0, 0), (0, 0)], 'constant', constant_values=0), np.tile(oc, [pad_len_a, 1, 1])])
            act = bimap_r(action_zeros, act, lambda a_z, a: np.concatenate([np.tile(a_z, [pad_len_b, a.shape[1]]+[1]*(len(a.shape) - 2)), a, np.tile(a_z, [pad_len_a, a.shape[1]]+[1]*(len(a.shape) - 2))]))
            rew = np.pad(rew, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            ret = np.pad(ret, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            emask = np.pad(emask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            tmask = np.pad(tmask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            omask = np.pad(omask, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
            lact = bimap_r(legal_actions_zeros, lact, lambda a_z, a: np.concatenate([np.tile(a_z, [pad_len_b, a.shape[1]]+[1]*(len(a.shape) - 2)), a, np.tile(a_z, [pad_len_a, a.shape[1]]+[1]*(len(a.shape) - 2))]))
            if(record_hidden_in and pad_len_b > 0):
                hidden_in = map_r(initial_hidden, lambda h: np.repeat(np.expand_dims(np.array(h),0),len(players),0))
            progress = np.pad(progress, [(pad_len_b, pad_len_a), (0, 0)], 'constant', constant_values=1)
            imi = np.pad(imi, [(pad_len_b, pad_len_a), (0, 0), (0, 0)], 'constant', constant_values=0)
        importance = np.full(omask.shape,ep.get('importance_weight',1.0))
        
        obss.append(obs)
        probs.append(prob)
        acts.append(act)
        lacts.append(lact)
        if(record_hidden_in):
            hidden_ins.append(hidden_in)
        datum.append((v, oc, rew, ret, emask, tmask, omask, progress, imi, importance))
        ep_indices.append((ep['ep_idx'],ep['train_start']))

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    prob = to_torch(bimap_r(prob_ones, rotate(probs), lambda _, p: np.array(p)))
    act = to_torch(bimap_r(action_zeros, rotate(acts), lambda _, a: np.array(a)))
    lact = to_torch(bimap_r(legal_actions_zeros, rotate(lacts), lambda _, a: np.array(a)))
    if(record_hidden_in):
        hidden_in = to_torch(bimap_r(initial_hidden, rotate(hidden_ins), lambda _, h: np.array(h)))
    else:
        hidden_in = None
    v, oc, rew, ret, emask, tmask, omask, progress, imi, importance = [to_torch(np.array(val)) for val in zip(*datum)]

    ret = {
        'observation': obs,
        'selected_prob': prob,
        'value': v,
        'action': act, 'outcome': oc,
        'reward': rew, 'return': ret,
        'episode_mask': emask,
        'turn_mask': tmask, 'observation_mask': omask,
        'legal_actions': lact,
        'progress': progress,
        'imitation': imi,
        'hidden_in': hidden_in,
        'ep_indices': ep_indices,
        'importance_weight': importance
    }
    return ret

def forward_prediction(model, hidden, batch, args):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: batch outputs of neural network
    """

    observations = batch['observation']  # (..., B, T, P or 1, ...)
    batch_shape = batch['observation_mask'].size()[:3]  # (B, T, P or 1)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.flatten(0, 2))  # (..., B * T * P or 1, ...)
        outputs = model(obs, None)
        outputs = map_r(outputs, lambda o: o.unflatten(0, batch_shape))  # (..., B, T, P or 1, ...)
    else:
        # sequential computation with RNN
        # RNN block is assumed to be an instance of ASRCAISim1.plugins.HandyRLUtility.RecurrentBlock.RecurrentBlock
        outputs = {}
        hidden_in=batch.pop('hidden_in',None)
        if hidden_in is not None:
            hidden = hidden_in

        seq_len = batch_shape[1]
        obs = map_r(observations, lambda o: o.transpose(1, 2))  # (..., B , P or 1 , T, ...)
        omask_ = map_r(batch['observation_mask'], lambda o: o.transpose(1, 2))  # (..., B , P or 1 , T, ...)
        omask = map_r(omask_, lambda h: omask_.view([*h.size()[:2], seq_len, *([1] * (h.dim() - 2))]).flatten(0,1))
        hidden_ = map_r(hidden, lambda h: h.flatten(0, 1))  # (..., B * P or 1, ...)
        if args['burn_in_steps'] >0 :
            model.eval()
            with torch.no_grad():
                outputs_ = model(map_r(obs, lambda o:o[:, :, :args['burn_in_steps'], ...].flatten(0, 2)), hidden_, args['burn_in_steps'],mask=map_r(omask,lambda o: o[:,:args['burn_in_steps'],...]))
            hidden_ = outputs_.pop('hidden')
            outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2], args['burn_in_steps'])).transpose(1, 2))
            for k, o in outputs_.items():
                outputs[k] = outputs.get(k, []) + [o]
        if not model.training:
            model.train()
        outputs_ = model(map_r(obs, lambda o:o[:, :, args['burn_in_steps']:, ...].flatten(0, 2)), hidden_,seq_len-args['burn_in_steps'],mask=map_r(omask,lambda o: o[:,args['burn_in_steps']:,...]))
        hidden_ = outputs_.pop('hidden')
        outputs_ = map_r(outputs_, lambda o: o.unflatten(0, (batch_shape[0], batch_shape[2], seq_len-args['burn_in_steps'])).transpose(1, 2))
        for k, o in outputs_.items():
            outputs[k] = outputs.get(k, []) + [o]
        outputs = {k: torch.cat(o, dim=1) for k, o in outputs.items() if o[0] is not None}

    for k, o in outputs.items():
        if k == 'policy':
            o = o.mul(batch['turn_mask'])
            if o.size(2) > 1 and batch_shape[2] == 1:  # turn-alternating batch
                o = o.sum(2, keepdim=True)  # gather turn player's policies
                outputs[k] = o
        else:
            # mask valid target values and cumulative rewards
            outputs[k] = o.mul(batch['observation_mask'])

    return outputs

@ray.remote
class BatcherWorker:
    def __init__(self, args, bid, queue, replay_buffer):
        self.args = args
        self.queue = queue
        self.replay_buffer = replay_buffer
        self.bid = bid
    def run(self):
        print('started batcher %d' % self.bid)
        while not ray.get(self.replay_buffer.is_ready.remote()):
            time.sleep(1)
        while True:
            episodes = ray.get([self.replay_buffer.select_episode.remote() for _ in range(self.args['batch_size'])])
            batch = make_batch(episodes, self.args)
            self.queue.put(batch)
        print('finished batcher %d' % self.bid)

class Batcher:
    def __init__(self, args, replay_buffer):
        self.args = args
        self.replay_buffer = replay_buffer
        queue_capacity = self.args.get('batch_queue_capacity',16)
        on_which_node = self.args.get('node_designation',None)
        if on_which_node is None:
            self.actor_options={}
        else:
            self.actor_options={'resources':{'node:{}'.format(on_which_node):0.001}}
        self.queue = Queue(maxsize=queue_capacity,actor_options=self.actor_options)

    def run(self):
        self.workers =  [BatcherWorker.options(num_cpus=1,**self.actor_options).remote(self.args,i,self.queue,self.replay_buffer) for i in range(self.args['num_batchers'])]
        for worker in self.workers:
            worker.run.remote()

    def batch(self):
        return self.queue.get()
