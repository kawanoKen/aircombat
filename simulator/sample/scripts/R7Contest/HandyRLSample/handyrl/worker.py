# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# worker and gather

import os
import random
from collections import defaultdict, deque, OrderedDict
import cloudpickle
import copy
import torch
import ray

import queue
from .environment import prepare_env, make_env
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper
from ASRCAISim1.plugins.HandyRLUtility.distribution import getActionDistributionClass

@ray.remote
class Worker:
    def __init__(self, args, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.exploration = None
        self.use_exploration = self.args.get("exploration_config",{}).get("use_exploration",False)
        if(self.use_exploration):
            self.exploration_cycle = 0
            self.exploration_config =  self.args["exploration_config"]
        self.learner = ray.get_actor(args['learner_name'])
        self.match_maker = ray.get_actor('MatchMaker')

        self.env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(self.env, self.args)
        self.evaluator = Evaluator(self.env, self.args)

        random.seed(args['seed'] + wid)
        self.numWeightsPerPolicy = 50
        self.custom_classes = self.args['custom_classes']
        self.model_pool = defaultdict(lambda:OrderedDict())
        self.model_epoch = -1
        for policyName,policyConfig in self.env.policy_config.items():
            model = self.get_model(policyName)
            model.model.load_state_dict(ray.get(self.match_maker.getLatestTrainingWeight.remote(policyName)), strict=True)
            model.eval()
            count=ray.get(self.match_maker.getTrainingWeightUpdateCount.remote(policyName))
            self.model_pool[policyName][0] = count, model

    def get_model(self,policyName):
        model_class = self.custom_classes[self.env.net(policyName)]
        obs_space = self.env.policy_config[policyName]['observation_space']
        ac_space = self.env.policy_config[policyName]['action_space']
        model_config = self.env.policy_config[policyName].get('model_config',{})
        if('actionDistributionClassGetter' in model_config):
            action_dist_class = self.custom_classes[model_config['actionDistributionClassGetter']](ac_space)
        else:
            action_dist_class = getActionDistributionClass(ac_space)
        model_config['custom_classes']=self.custom_classes
        model_config['auxiliary_task_truth_space']=self.env.policy_config[policyName]['auxiliary_task_truth_space']
        return ModelWrapper(model_class(obs_space, ac_space, action_dist_class, model_config))

    def load_weight(self,policyName,weight_id):
        if(weight_id<=0):
            count=ray.get(self.match_maker.getTrainingWeightUpdateCount.remote(policyName))
            model = self.model_pool[policyName][0][1]
            if count > self.model_pool[policyName][0][0]:
                model.model.load_state_dict(ray.get(self.match_maker.getLatestTrainingWeight.remote(policyName)), strict=True)
                self.model_pool[policyName][0] = count, model
        elif weight_id in self.model_pool[policyName]:
            model = self.model_pool[policyName].pop(weight_id)
            self.model_pool[policyName][weight_id] = model
        else:
            model = self.get_model(policyName)
            weight = ray.get(self.match_maker.loadWeight.remote(policyName, weight_id))
            model.model.load_state_dict(weight, strict=True)
            self.model_pool[policyName][weight_id] = model
            if(len(self.model_pool[policyName])>self.numWeightsPerPolicy):
                #メモリに保持している重みが上限を超えたら直近の読み込み時刻が最も古いもの(ただし、0を除く)を削除する。
                it=iter(self.model_pool[policyName].keys())
                removed=next(it)
                if(removed==0):
                    removed=next(it)
                self.model_pool[policyName].pop(removed,None)
        if model.training:
            model.eval()
        return model

    def _gather_models(self, match_info):
        models_for_policies = {}
        for team, info in match_info.get("teams",match_info).items():
            models_for_policies[info["Policy"]+info["Suffix"]] = self.load_weight(info["Policy"], info["Weight"])
            if(info["Policy"] in self.args["policy_to_imitate"]):
                models_for_policies["Imitator"]=self.load_weight(self.args['policy_to_train'], -1)
        return models_for_policies

    def run(self):
        while True:
            args = ray.get(self.learner.make_match.remote(self.worker_id))
            if args is None:
                break
            role = args['role']
            self.model_epoch = args['model_epoch']
            models_for_policies = self._gather_models(args['match_info'])
            args['deterministics']={
                info["Policy"]+info["Suffix"]: info['deterministic']
                for info in args['match_info'].get('teams',args['match_info']).values()
            }

            if role == 'g':
                if(self.use_exploration):
                    alpha = self.exploration_config.get('alpha',0.7)
                    decay = self.exploration_config.get('eps_decay',-1)
                    if(decay <= 0):
                        eps = self.exploration_config.get('eps',self.exploration_config.get('eps_start',0.4))
                    else:
                        eps_start = self.exploration_config.get('eps_start',1.0)
                        eps_end = self.exploration_config.get('eps_end',0.1)
                        eps = eps_start + (eps_end - eps_start) * min(1.0,self.model_epoch / decay)
                    cycle = int(self.exploration_config.get('cycle',1))
                    self.exploration_cycle = (self.exploration_cycle + 1) % cycle
                    self.exploration = eps**(1+1.0*((self.worker_id-1)*cycle+self.exploration_cycle)/(cycle*self.args['worker']['num_parallel']-1)*alpha)
                args['exploration'] = {
                    info["Policy"]+info["Suffix"]: self.exploration if info['Policy'] == self.args['policy_to_train'] and info['Weight'] < 0 else None
                    for info in args['match_info'].get('teams',args['match_info']).values()
                }
                if(self.args.get('replay_buffer_config',{}).get('calc_initial_priority',False)):
                    args['model_for_eval']=self.load_weight(self.args['policy_to_train'],0)
                episode = self.generator.execute(models_for_policies, args)
                self.learner.feed_episodes.remote([episode])
            elif role == 'e':
                result = self.evaluator.execute(models_for_policies, args)
                self.learner.feed_results.remote([result])
        self.learner.notifyWorkerShutdown.remote(self.worker_id)
        print('closed worker %d' % self.worker_id)
        ray.actor.exit_actor()

    def is_alive(self):
        return True
