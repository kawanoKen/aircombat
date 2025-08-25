# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# episode generation

import random
import bz2
import cloudpickle
import collections

import numpy as np
from gymnasium import spaces
import torch

from .util import softmax, map_r
from .batcher import make_batch, forward_prediction
from .losses import compute_target

class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        if(hasattr(env,"env")):
            #unwrap env
            self.match_monitor = args['custom_classes'][args["match_monitor_class"]](env.env)
        else:
            self.match_monitor = args['custom_classes'][args["match_monitor_class"]](env)
    def generate(self, models_for_policies, args):
        # episode generation
        moments = []
        models = {}
        hidden = {}
        deterministics={}
        exploration={}
        #replace config of env
        self.match_monitor.onEpisodeBegin()
        self.env.setMatch(args['match_info'])

        err = self.env.reset()
        if err:
            return None

        only_single_explorer_per_policy=self.args.get('exploration_config',{}).get('only_one_explorer_per_policy',False)
        if only_single_explorer_per_policy:
            player_to_policy = collections.defaultdict(lambda: [])
            for player in self.env.players():
                exploration[player] = False
                player_to_policy[self.env.policy_map[player]].append(player)
            for policy, players in player_to_policy.items():
                if policy != "Imitator":
                    explorer = random.choice(players)
                    exploration[explorer] = True
        else:
            for player in self.env.players():
                policy = self.env.policy_map[player]
                if policy != "Imitator":
                    exploration[player] = args['exploration'][policy]

        #記録対象の選別
        players_to_record=self.env.players()
        if self.args.get('replay_buffer_config',{}).get('discard_untrainable_players',False):
            players_to_record = [player for player in players_to_record
                if self.env.policy_map[player] in [self.args['policy_to_train']] + ['Imitator']
            ]

        while not self.env.terminal():
            moment_keys = ['observation', 'selected_prob', 'legal_actions', 'action', 'value', 'reward', 'return', 'hidden_in']
            moment = {key: {p: None for p in players_to_record} for key in moment_keys}
            actions = {p: None for p in self.env.players()}

            turn_players = self.env.turns()
            observers = self.env.observers()
            for player in self.env.players():
                if player not in turn_players and (player not in observers or not self.args['observation']):
                    continue

                obs = self.env.observation(player)
                if not player in models:
                    models[player] = models_for_policies[self.env.policy_map[player]]
                    hidden[player] = models[player].init_hidden()
                    deterministics[player] = args['deterministics'][self.env.policy_map[player]]

                model = models[player]
                hidden_in = hidden[player]
                outputs = model.inference(obs, hidden[player])
                hidden[player] = outputs.get('hidden', None)
                v = outputs.get('value', None)

                if player in players_to_record:
                    moment['observation'][player] = obs
                    moment['value'][player] = v
                    if self.args.get('record_hidden_in',True):
                        moment['hidden_in'][player] = hidden_in

                if player in turn_players:
                    legal_actions_ = self.env.legal_actions(player)
                    dist = models[player].get_action_dist(outputs['policy'], legal_actions_)
                    action_ = dist.sample(deterministics[player],exploration[player])
                    selected_prob_ = dist.log_prob(action_)
                    actions[player] = action_
                    if player in players_to_record:
                        moment['selected_prob'][player] = selected_prob_
                        moment['legal_actions'][player] = legal_actions_
                        moment['action'][player] = action_
            err = self.env.step(actions)
            if err:
                return None
            
            #教師役として用いる場合、模倣する側の情報を付加
            imitationInfo = self.env.getImitationInfo()
            for player,info in imitationInfo.items():
                if not player in models:
                    models[player] = models_for_policies[self.env.policy_map[player]] #"Imitator"
                    hidden[player] = models[player].init_hidden()
                    deterministics[player] = True
                model = models[player]
                outputs = model.inference(info['observation'], hidden[player])
                moment['hidden_in'][player] = hidden[player]
                moment['observation'][player] = info['observation']
                moment['action'][player] = info['action']
                moment['selected_prob'][player] = map_r(info['action'], lambda x: np.zeros_like(x,dtype=np.float32))
                moment['value'][player] = outputs.get('value', None)
                hidden[player] = outputs.get('hidden', None)
                moment['legal_actions'][player] = info['legal_actions']

            reward = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = reward.get(player, None)

            moment['turn'] = turn_players + list(imitationInfo.keys())
            moments.append(moment)

        if len(moments) < 1:
            return None

        #補助タスクの正解データを抽出
        aux_truth=self.env.getAuxiliaryTaskTruth()
        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret
        args['player']=self.env.players()
        if(self.args.get('replay_buffer_config',{}).get('calc_initial_priority',False)):
            model_for_eval = args.pop('model_for_eval')
        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'score': self.env.score(),
            'moment': [
                bz2.compress(cloudpickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ],
            'policy_map': self.env.policy_map,
            'match_maker_result': self.match_monitor.onEpisodeEnd(args['match_type']),
            'auxiliary_task_truth': aux_truth,
        }
        if(self.args.get('replay_buffer_config',{}).get('calc_initial_priority',False)):
            target_errors, advantages = self.calc_target_errors_and_advantages_for_priority(episode, model_for_eval)
            episode['target_errors_for_priority'] = target_errors
            episode['advantages_for_priority'] = advantages

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode

    def calc_target_errors_and_advantages_for_priority(self, ep, model_for_eval):
        #Prioritized replayの優先度を計算するためのtarget_errorとadvantageを計算する。

        #ダミーバッチの生成
        new_args = {k: v for k, v in self.args.items()}
        new_args['forward_steps'] = 0
        ep_idx = 0
        new_args['forward_steps'] = max(new_args['forward_steps'], ep['steps'])
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

        with torch.no_grad():
            batch = make_batch(ep_for_dummy_batch, new_args)
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = model_for_eval.init_hidden([batch_size, player_count])
            ep_indices = batch.pop("ep_indices")
            outputs = forward_prediction(model_for_eval, hidden, batch, self.args)
            if hidden is not None and self.args['burn_in_steps'] > 0:
                batch = map_r(batch, lambda v: v[:, self.args['burn_in_steps']:] if v.size(1) > 1 else v)
                outputs = map_r(outputs, lambda v: v[:, self.args['burn_in_steps']:])
            batch["ep_indices"]=ep_indices

        #targetとadvantageの計算
        with torch.no_grad():
            dist = model_for_eval.get_action_dist(outputs['policy'], batch['legal_actions'])
            log_probs = dist.unpack_scalar(dist.log_prob(batch['action']),keepDim=True)
            actions = dist.unpack_scalar(batch['action'],keepDim=True)
            selected_probs = dist.unpack_scalar(batch['selected_prob'],keepDim=True)
            entropies = dist.unpack_scalar(dist.entropy(),keepDim=True)
            separate_policy_gradients = self.args.get('separate_policy_gradients',True)
            if not separate_policy_gradients:
                log_probs = [sum(log_probs)]
                selected_probs = [sum(selected_probs)]
                entropies = [sum(entropies)]
            emasks = batch['episode_mask']
            clip_rho_threshold, clip_c_threshold = 1.0, 1.0

            outputs_nograd = {k: o.detach() for k, o in outputs.items()}
            if 'value' in outputs_nograd:
                values_nograd = outputs_nograd['value']
                if self.args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
                    values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
                    values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
                outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

            losses_total = {}
            target_errors_for_priority = {}
            advantages_for_priority = {}
            for idx in range(len(log_probs)):
                log_selected_b_policies = selected_probs[idx] * emasks
                log_selected_t_policies = log_probs[idx] * emasks

                # thresholds of importance sampling
                log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
                rhos = torch.exp(log_rhos)
                clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
                cs = torch.clamp(rhos, 0, clip_c_threshold)

                # compute targets and advantage
                targets = {}
                advantages = {}

                value_args = outputs_nograd.get('value', None), batch['outcome'], None, self.args['lambda'], 1, clipped_rhos, cs
                return_args = outputs_nograd.get('return', None), batch['return'], batch['reward'], self.args['lambda'], self.args['gamma'], clipped_rhos, cs

                prio_target = self.args.get('replay_buffer_config', {}).get('priority_target','TD')
                if(prio_target=='SAME'):
                    targets['value'], advantages['value'] = compute_target(self.args['value_target'], *value_args)
                    targets['return'], advantages['return'] = compute_target(self.args['value_target'], *return_args)
                    if self.args['policy_target'] != self.args['value_target']:
                        _, advantages['value'] = compute_target(self.args['policy_target'], *value_args)
                        _, advantages['return'] = compute_target(self.args['policy_target'], *return_args)
                else:
                    targets['value'], advantages['value'] = compute_target(prio_target, *value_args)
                    targets['return'], advantages['return'] = compute_target(prio_target, *return_args)

                tmasks = batch['turn_mask']
                omasks = batch['observation_mask']
                if 'value' in outputs:
                    target_errors_for_priority['value'+str(idx)] = (outputs['value']-targets['value']).mul(omasks).detach().to('cpu')
                if 'return' in outputs:
                    target_errors_for_priority['return'+str(idx)] = (outputs['return']-targets['return']).mul(omasks).detach().to('cpu')
                advantages_for_priority['value'+str(idx)] = advantages['value'].mul(tmasks).detach().to('cpu')
                advantages_for_priority['return'+str(idx)] = advantages['return'].mul(tmasks).detach().to('cpu')
        return target_errors_for_priority, advantages_for_priority
