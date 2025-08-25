# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# training

import os
import time
import datetime
import copy
import threading
import random
import bz2
import cloudpickle
import warnings
import queue
from collections import deque

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import psutil
import ray

from .environment import prepare_env, make_env
from .util import map_r, bimap_r, trimap_r, rotate
from .model import to_torch, to_gpu, ModelWrapper
from .losses import compute_target
from .worker import Worker
from .batcher import replace_none, make_batch, forward_prediction, Batcher
from ASRCAISim1.plugins.HandyRLUtility.distribution import getActionDistributionClass
from ASRCAISim1.plugins.Supervised.AuxiliaryTaskUtilForHandyRL import AuxiliaryTaskManager
from .ReplayBuffer import DefaultReplayBuffer, PrioritizedReplayBuffer

def compose_losses(outputs, entropies, log_selected_policies, clipped_rhos, summed_advantages, targets, batch, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    tmasks = batch['turn_mask']
    omasks = batch['observation_mask']
    importance = batch['importance_weight']
    tmasks = tmasks.mul(importance)
    omasks = omasks.mul(importance)
    # imitation masking
    imitation_mask = batch['imitation']
    contains_imitation_data = imitation_mask.sum()>0
    if(contains_imitation_data and args.get("disable_rl_from_imitator",False)):
        imitation_mask_for_rl = 1.0-imitation_mask
        tmasks_for_rl = tmasks.mul(imitation_mask_for_rl)
        omasks_for_rl = omasks.mul(imitation_mask_for_rl)
    else:
        tmasks_for_rl = tmasks
        omasks_for_rl = omasks
    tmasks_for_imitation = tmasks.mul(imitation_mask)
    omasks_for_imitation = omasks.mul(imitation_mask)

    losses = {}
    dcnt = tmasks.sum().item()
    clipped_advantages = clipped_rhos * summed_advantages
    losses['p'] = (-log_selected_policies * clipped_advantages).mul(tmasks_for_rl).sum()
    if 'value' in outputs:
        losses['v'] = ((outputs['value'] - targets['value']) ** 2).mul(omasks_for_rl).sum() / 2
    if 'return' in outputs:
        losses['r'] = F.smooth_l1_loss(outputs['return'], targets['return'], reduction='none').mul(omasks_for_rl).sum()

    entropy = entropies.mul(tmasks_for_rl.sum(-1,keepdim=True))
    losses['ent'] = entropy.sum()

    base_loss = losses['p'] + losses.get('v', 0) + losses.get('r', 0) + losses.get('r_i', 0)
    entropy_loss = entropy.mul(1 - batch['progress'].unsqueeze(-1) * (1 - args['entropy_regularization_decay'])).sum() * -args['entropy_regularization']
    losses['ent_loss']=entropy_loss
    losses['total'] = base_loss + entropy_loss

    # imitation loss
    if(contains_imitation_data):
        imitation_beta = args.get('imitation_beta', 1.0)
        if('imitation_kl_threshold' in args):
            clip_imitation_kl_threshold = args['imitation_kl_threshold']
            clip_imitation_loss_threshold = args.get('imitation_loss_threshold', clip_imitation_kl_threshold)
        elif('imitation_loss_threshold' in args):
            clip_imitation_loss_threshold = args['imitation_loss_threshold']
            clip_imitation_kl_threshold = args.get('imitation_kl_threshold', clip_imitation_loss_threshold)
        else:
            clip_imitation_kl_threshold = 10.0
            clip_imitation_loss_threshold = 10.0
        imitation_loss_scale=args.get('imitation_loss_scale',1.0)
        if(imitation_beta > 0.0):
            args["imitation_adv_ma"]+=args["imitation_adv_ma_update_rate"]*(float(torch.mean(torch.pow(summed_advantages, 2.0)))-args["imitation_adv_ma"])
            kl = -(torch.exp(imitation_beta*summed_advantages/(1e-8+pow(args["imitation_adv_ma"],0.5))).detach() * log_selected_policies).mul(tmasks_for_imitation)
        else:
            kl = -log_selected_policies.mul(tmasks_for_imitation)
        kl = torch.clamp(kl,0,clip_imitation_kl_threshold)
        kl = torch.clamp(kl*imitation_loss_scale, 0, clip_imitation_loss_threshold).sum()
        losses['imi'] = kl
        losses['total'] += kl

    return losses, dcnt

def compute_loss(batch, model, hidden, args, replay_buffer):
    ep_indices = batch.pop("ep_indices")
    outputs = forward_prediction(model, hidden, batch, args)
    if hidden is not None and args['burn_in_steps'] > 0:
        batch = map_r(batch, lambda v: v[:, args['burn_in_steps']:] if v.size(1) > 1 else v)
        outputs = map_r(outputs, lambda v: v[:, args['burn_in_steps']:])
    batch["ep_indices"]=ep_indices
    dist = model.get_action_dist(outputs['policy'], batch['legal_actions'])
    actions = dist.unpack_scalar(batch['action'],keepDim=True)
    log_probs = dist.unpack_scalar(dist.log_prob(batch['action']),keepDim=True)
    selected_probs = dist.unpack_scalar(batch['selected_prob'],keepDim=True)
    entropies = dist.unpack_scalar(dist.entropy(batch['action']),keepDim=True)
    separate_policy_gradients = args.get('separate_policy_gradients',True)
    if not separate_policy_gradients:
        log_probs = [sum(log_probs)]
        selected_probs = [sum(selected_probs)]
        entropies = [sum(entropies)]
    emasks = batch['episode_mask']
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    outputs_nograd = {k: o.detach() for k, o in outputs.items()}

    if 'value' in outputs_nograd:
        values_nograd = outputs_nograd['value']
        if args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
        outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

    # calculate losses for each action component
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

        value_args = outputs_nograd.get('value', None), batch['outcome'], None, args['lambda'], 1, clipped_rhos, cs
        return_args = outputs_nograd.get('return', None), batch['return'], batch['reward'], args['lambda'], args['gamma'], clipped_rhos, cs

        targets['value'], advantages['value'] = compute_target(args['value_target'], *value_args)
        targets['return'], advantages['return'] = compute_target(args['value_target'], *return_args)

        if args['policy_target'] != args['value_target']:
            _, advantages['value'] = compute_target(args['policy_target'], *value_args)
            _, advantages['return'] = compute_target(args['policy_target'], *return_args)

        tmasks = batch['turn_mask']
        omasks = batch['observation_mask']

        prio_target = args.get('replay_buffer_config', {}).get('priority_target','TD')
        if(prio_target=='SAME'):
            if 'value' in outputs:
                target_errors_for_priority['value'+str(idx)] = (outputs['value']-targets['value']).mul(omasks).detach().to('cpu')
            if 'return' in outputs:
                target_errors_for_priority['return'+str(idx)] = (outputs['return']-targets['return']).mul(omasks).detach().to('cpu')
            advantages_for_priority['value'+str(idx)] = advantages['value'].mul(tmasks).detach().to('cpu')
            advantages_for_priority['return'+str(idx)] = advantages['return'].mul(tmasks).detach().to('cpu')
        else:
            tfp_v, afp_v = compute_target(prio_target, *value_args)
            tfp_r, afp_r = compute_target(prio_target, *return_args)
            if 'value' in outputs:
                target_errors_for_priority['value'+str(idx)] = (outputs['value']-tfp_v).mul(omasks).detach().to('cpu')
            if 'return' in outputs:
                target_errors_for_priority['return'+str(idx)] = (outputs['return']-tfp_r).mul(omasks).detach().to('cpu')
            advantages_for_priority['value'+str(idx)] = afp_v.mul(tmasks).detach().to('cpu')
            advantages_for_priority['return'+str(idx)] = afp_r.mul(tmasks).detach().to('cpu')

        # compute policy advantage
        summed_advantages = sum(advantages.values())
        losses, dcnt = compose_losses(outputs, entropies[idx], log_selected_t_policies, clipped_rhos, summed_advantages, targets, batch, args)
        for k,v in losses.items():
            losses_total[k] = losses_total.get(k,0) + v

    replay_buffer.update.remote(batch['ep_indices'], target_errors_for_priority, advantages_for_priority)
    return losses_total, dcnt

class Trainer:
    def __init__(self, args, model, summary_writer):
        self.args = args
        if torch.cuda.is_available():
            self.gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.gpu = 1
        else:
            self.gpu = 0
        self.model = model
        self.summary_writer = summary_writer
        self.args['initial_hidden'] = self.model.init_hidden()
        self.args['observation_space'] = self.model.observation_space
        self.args['action_space'] = self.model.action_space
        self.args['auxiliary_task_truth_space'] = self.model.model_config['auxiliary_task_truth_space']
        self.args['action_dist_class'] = self.model.action_dist_class
        self.auto_tune_lr = self.args.get('auto_tune_lr',True)
        if self.auto_tune_lr:
            self.default_lr = self.args.get('default_lr',3e-8)
            self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
            lr = self.default_lr * self.data_cnt_ema
        else:
            lr = self.args.get('lr',1e-4)
        self.params = list(self.model.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        if self.args.get('prioritized_replay',True):
            replayBufferClass = PrioritizedReplayBuffer
        else:
            replayBufferClass = DefaultReplayBuffer
        on_which_node = self.args.get('replay_buffer_config',{}).get('node_designation',None)
        if on_which_node is None:
            remoteReplayBufferClass = ray.remote(num_cpus=1)(replayBufferClass)
        else:
            remoteReplayBufferClass = ray.remote(num_cpus=1,resources={"node:{}".format(on_which_node): 0.001})(replayBufferClass)
        self.replay_buffer = remoteReplayBufferClass.remote(self.args)
        self.batcher = Batcher(self.args, self.replay_buffer)
        self.batcher.run()
        self.update_flag = False
        self.update_queue = queue.Queue(maxsize=1)

        self.wrapped_model = ModelWrapper(self.model)
        self.trained_model = self.wrapped_model
        self.model_for_eval = copy.deepcopy(self.model)
        self.model_for_eval.cpu()
        self.model_for_eval.eval()
        if self.gpu > 1:
            self.trained_model = nn.DataParallel(self.wrapped_model)
        # imitation
        self.args["imitation_adv_ma_initial"]=self.args.get("imitation_adv_ma_initial",100.0)
        self.args["imitation_adv_ma_update_rate"]=self.args.get("imitation_adv_ma_update_rate",1.0e-8)
        self.args["imitation_adv_ma"]=self.args["imitation_adv_ma_initial"]
        self.update_count = 0
        self.reset_weight_queue = queue.Queue(maxsize=1)

        #auxiliary task
        self.use_auxiliary_tasks = len(self.args.get('auxiliary_tasks',{})) > 0
        if self.use_auxiliary_tasks:
            self.auxiliary_tasks = AuxiliaryTaskManager(self.args,model)

    def request_to_reset_weight(self, initial_weight):
        try:
            self.reset_weight_queue.put_nowait(initial_weight)
        except queue.Full:
            pass

    def update(self):
        self.update_flag = True
        model, steps = self.update_queue.get()
        self.update_count += 1
        return model, steps

    def train(self):
        if self.optimizer is None:  # non-parametric model
            time.sleep(0.1)
            return self.model

        batch_cnt, data_cnt, loss_sum = 0, 0, {}
        if self.use_auxiliary_tasks:
            loss_aux_sum = {}
        if self.gpu > 0:
            if torch.cuda.is_available():
                self.trained_model.cuda()
            elif torch.backends.mps.is_available():
                self.trained_model.to('mps')
        self.trained_model.train()

        while data_cnt == 0 or not self.update_flag:
            batch = self.batcher.batch()
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = self.wrapped_model.init_hidden([batch_size, player_count])
            ep_indices = batch.pop("ep_indices")
            if self.gpu > 0:
                batch = to_gpu(batch)
                hidden = to_gpu(hidden)
            batch["ep_indices"]=ep_indices

            losses, dcnt, = compute_loss(batch, self.trained_model, hidden, self.args, self.replay_buffer)

            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.step()

            batch_cnt += 1
            data_cnt += dcnt
            for k, l in losses.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + l.item()

            if self.use_auxiliary_tasks:
                loss_aux,metrics_aux=self.auxiliary_tasks.train()
                for k, l in loss_aux.items():
                    loss_aux_sum[k] = loss_aux_sum.get(k,0.0) + l.item()

            self.steps += 1

        for k, l in loss_sum.items():
            self.summary_writer.add_scalar("trainer/loss/"+k,l/data_cnt,self.update_count)
        if self.use_auxiliary_tasks:
            for k, l in loss_aux_sum.items():
                self.summary_writer.add_scalar("trainer/loss/auxiliary/"+k,l,self.update_count)
        self.summary_writer.add_scalar("trainer/num_updates",self.steps,self.update_count)
        if self.use_auxiliary_tasks:
            if(len(metrics_aux)>0):
                for task_name,metrics in metrics_aux.items():
                    for k,m in metrics.items():
                        self.summary_writer.add_scalar("trainer/auxiliary/"+task_name+"/"+k,m,self.update_count)
        for k,m in ray.get(self.replay_buffer.get_metrics.remote()).items():
            self.summary_writer.add_scalar("trainer/replay_buffer/"+k,m,self.update_count)
        self.summary_writer.flush()
        print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))
        if self.use_auxiliary_tasks:
            if len(loss_aux_sum)>0:
                print('loss(aux) = %s' % ' '.join([k + ':' + '%.3f' % l for k, l in loss_aux_sum.items()]))
        if self.use_auxiliary_tasks:
            if(len(metrics_aux)>0):
                for task_name,metrics in metrics_aux.items():
                    print("metrics("+task_name+")")
                    for k,m in metrics.items():
                        print("    "+k+":",m)
        self.summary_writer.add_scalar("trainer/num_updates",self.steps,self.update_count)
        if self.auto_tune_lr:
            self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.default_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        self.model.cpu()
        self.model.eval()
        reset_weight = None
        while True:
            try:
                reset_weight = self.reset_weight_queue.get_nowait()
            except queue.Empty:
                break
        if reset_weight is not None:
            print("============reset weight========")
            self.model.load_state_dict(reset_weight, strict=False)
        self.model_for_eval.load_state_dict(self.model.state_dict())

    def run(self):
        print(self.args['learner_name'],'waiting training')
        while not ray.get(self.replay_buffer.is_ready.remote()):
            time.sleep(1)
        print(self.args['learner_name'],'started training')
        while True:
            self.train()
            self.update_flag = False
            self.update_queue.put((self.model_for_eval, self.steps))
        print(self.args['learner_name'],'finished training')

    def add_episodes(self,episodes):
        self.replay_buffer.add_episodes.remote(episodes)

class Learner:
    def __init__(self, args, net=None, remote=False):
        self.name = args['name']
        self.custom_classes = args['custom_classes']
        #保存先の設定
        self.save_dir = args['save_dir']
        #SummaryWriterの生成
        self.summary_writer = SummaryWriter(os.path.join(self.save_dir,'logs',self.name))
        #引数の修正
        train_args = args['train_args']
        train_args['policy_to_imitate']=train_args.get('policy_to_imitate',[])
        env_args = args['env_args']
        env_args['policy_config'] = args['policy_config']
        env_args['teams'] = args['teams']
        train_args['env'] = env_args
        train_args['custom_classes']=args['custom_classes']
        train_args['learner_name']=self.name
        self.args = train_args
        worker_options=self.args.get("worker_options",{})
        self.workers = {i+1: Worker.options(name=self.name+":Worker_"+str(i),**worker_options).remote(self.args,i+1) for i in range(self.args.get("worker",{}).get("num_parallel",1))}

        random.seed(self.args['seed'])

        self.env = make_env(env_args)
        self.eval_rate = self.args['eval_rate']
        self.shutdown_flag = False
        self.flags = set()

        #MatchMakerの読み込み
        self.match_maker=ray.get_actor('MatchMaker')

        # trained datum
        self.policy_to_train = self.args['policy_to_train']
        self.model_epoch = 0
        self.models = {policyName: self.get_model(policyName)
            for policyName in self.env.policy_config.keys()
        }

        #初期重みの読み込みとpoolへの追加
        populate_config = ray.get(self.match_maker.checkInitialPopulation.remote())
        def saver(state_dict,dstPath):
            os.makedirs(os.path.dirname(dstPath), exist_ok=True)
            torch.save(state_dict, dstPath)
        def loader(srcPath):
            return torch.load(srcPath, map_location=torch.device('cpu'))
        for policyName, policyConfig in self.env.policy_config.items():
            if policyName == self.policy_to_train:
                self.match_maker.registerSaverLoader.remote(policyName, saver, loader, ".pth")
                model = self.get_model(policyName)
                initial_weight = ray.get(self.match_maker.getInitialWeight.remote(policyName))
                if(initial_weight is not None):
                    model.load_state_dict(initial_weight, strict = False)
                self.match_maker.uploadTrainingWeights.remote({policyName:model.state_dict()})
                self.match_maker.saveWeight.remote(policyName, 0, model.state_dict())
                self.models[policyName] = model
                if(policyName in populate_config):
                    self.populate(policyName, populate_config[policyName]['weight_id'], populate_config[policyName]['reset'])

        # generated datum
        self.generation_results = {}
        self.num_episodes = 0
        self.num_returned_episodes = 0

        # evaluated datum
        self.results = {}
        self.results_per_opponent = {}
        self.num_results = 0
        self.num_returned_results = 0

        self.prev_update_episodes = self.args['minimum_episodes']
        # no update call before storing minimum number of episodes + 1 epoch
        self.next_update_episodes = self.prev_update_episodes + self.args['update_episodes']
        self.checkpoint_interval = self.args.get('checkpoint_interval',1)

        # thread connection
        self.trainer = Trainer(self.args, self.models[self.policy_to_train], self.summary_writer)
        self.models[self.policy_to_train] = self.trainer.model_for_eval
        # open training thread
        threading.Thread(target=self.trainer.run, daemon=True).start()
        self.use_auxiliary_tasks = len(self.args.get('auxiliary_tasks',{})) > 0

    def run(self):
        self.startedTime = time.time()
        for worker in self.workers.values():
            worker.run.remote()

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
        return model_class(obs_space, ac_space, action_dist_class, model_config)

    def model_path(self, policyName, model_id):
        return os.path.join(self.save_dir, 'policies', 'checkpoints', policyName+'-'+str(model_id) + '.pth')

    def latest_model_path(self, policyName):
        return os.path.join(self.save_dir, 'policies', 'checkpoints', policyName+'-latest.pth')

    def match_maker_checkpoint_path(self, epoch):
        return os.path.join(self.save_dir, 'matches', 'checkpoints', self.name+'-'+str(epoch) + '.dat')

    def setup_initial_models(self):
        #call from at least one Learner after all Learners are created.
        for policyName, policyConfig in self.env.policy_config.items():
            count=ray.get(self.match_maker.getTrainingWeightUpdateCount.remote(policyName))
            if count<0:
                def saver(state_dict,dstPath):
                    os.makedirs(os.path.dirname(dstPath), exist_ok=True)
                    torch.save(state_dict, dstPath)
                def loader(srcPath):
                    return torch.load(srcPath, map_location=torch.device('cpu'))
                self.match_maker.registerSaverLoader.remote(policyName, saver, loader, ".pth")
                model = self.get_model(policyName)
                initial_weight = ray.get(self.match_maker.getInitialWeight.remote(policyName))
                if(initial_weight is not None):
                    model.load_state_dict(initial_weight, strict = False)
                self.match_maker.uploadTrainingWeights.remote({policyName:model.state_dict()})
                self.match_maker.saveWeight.remote(policyName, 0, model.state_dict())
    
    def populate(self, policyName, weight_id, reset):
        if(policyName==self.policy_to_train):
            fileName = policyName + "-" + str(weight_id) + ".pth"
            self.match_maker.saveWeight.remote(policyName, weight_id, self.models[policyName].state_dict())
            if reset:
                initial_weight = ray.get(self.match_maker.getInitialWeight.remote(policyName))
                if(initial_weight is None):
                    dummy = self.get_model(policyName)
                    initial_weight = dummy.state_dict()
                self.models[policyName].load_state_dict(initial_weight,strict=False)
                self.trainer.request_to_reset_weight(initial_weight)
            print("=====weight populated===== ",policyName," -> ",fileName, ("(reset)" if reset else ""))

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_epoch += 1
        self.models[self.policy_to_train] = model
        self.match_maker.uploadTrainingWeights.remote({self.policy_to_train:model.state_dict()})
        torch.save(model.state_dict(), self.latest_model_path(self.policy_to_train))
        if self.model_epoch % self.checkpoint_interval == 0:
            torch.save(model.state_dict(), self.model_path(self.policy_to_train,self.model_epoch))
            self.match_maker.save.remote(self.match_maker_checkpoint_path(self.model_epoch))

    def feed_episodes(self, episodes):
        # analyze generated episodes
        def get_policy_label_for_log(info):
            ret = info['Policy']
            if info['Suffix'] != '':
                weight_id = info['Weight']
                if weight_id == 0:
                    ret = ret + "*"
                elif weight_id > 0:
                    ret = ret + str(weight_id)
            return ret

        for episode in episodes:
            if episode is None:
                continue
            teamInfo=episode['args']['match_info'].get('teams',episode['args']['match_info'])
            for team, info in teamInfo.items():
                policyName=info['Policy']#+info['Suffix']
                weight_id=info['Weight']
                outcome = np.mean([episode['outcome'][p] for p in episode['args']['player']
                    if episode['policy_map'][p] == info['Policy']+info['Suffix']
                ])
                score = np.mean([episode['score'][p] for p in episode['args']['player']
                    if episode['policy_map'][p] == info['Policy']+info['Suffix']
                ])
                moments_ = sum([cloudpickle.loads(bz2.decompress(ms)) for ms in episode['moment']], [])
                rewards = [
                    np.array([replace_none(m['reward'][p], [0]) for m in moments_], dtype=np.float32) for p in episode['args']['player']
                    if episode['policy_map'][p] == info['Policy']+info['Suffix']
                ]
                reward_mean = np.mean(rewards)
                reward_total = np.sum(rewards)
                if(policyName==self.policy_to_train and weight_id<0):
                    model_epoch = episode['args']['model_epoch']
                    n, r, r2 = self.generation_results.get(model_epoch, (0, 0, 0))
                    self.generation_results[model_epoch] = n + 1, r + score, r2 + score ** 2

                policyLabelForLog = get_policy_label_for_log(info)
                self.summary_writer.add_scalar("generation/outcome/"+policyLabelForLog,outcome,self.num_returned_episodes)
                self.summary_writer.add_scalar("generation/score/"+policyLabelForLog,score,self.num_returned_episodes)
                self.summary_writer.add_scalar("generation/reward_mean/"+policyLabelForLog,reward_mean,self.num_returned_episodes)
                self.summary_writer.add_scalar("generation/reward_total/"+policyLabelForLog,reward_total,self.num_returned_episodes)

            populate_config = ray.get(self.match_maker.onEpisodeEnd.remote(episode['args']['match_info'],episode['match_maker_result']))
            for policyName,conf in populate_config.items():
                self.populate(policyName, conf['weight_id'], conf['reset'])
            match_maker_metrics = ray.get(self.match_maker.get_metrics.remote(episode['args']['match_info'],episode['match_maker_result']))
            for k,v in match_maker_metrics.items():
                self.summary_writer.add_scalar("match_maker/"+k,v,self.num_returned_episodes+self.num_returned_results)
            self.num_returned_episodes += 1

        # store generated episodes
        self.trainer.add_episodes([e for e in episodes if e is not None])
        if self.use_auxiliary_tasks:
            self.trainer.auxiliary_tasks.add_episodes([e for e in episodes if e is not None])

        mem_percent = psutil.virtual_memory().percent
        self.summary_writer.add_scalar("trainer/learner/mem_percent",mem_percent,self.num_returned_episodes)
        self.summary_writer.flush()

        if self.num_returned_episodes >= self.next_update_episodes:
            self.prev_update_episodes = self.next_update_episodes
            self.next_update_episodes = self.prev_update_episodes + self.args['update_episodes']
            self.update()
            if self.args['epochs'] >= 0 and self.model_epoch >= self.args['epochs']:
                self.shutdown_flag = True

    def feed_results(self, results):
        # store evaluation results
        def get_policy_label_for_log(info):
            ret = info['Policy']
            if info['Suffix'] != '':
                weight_id = info['Weight']
                if weight_id == 0:
                    ret = ret + "*"
                elif weight_id > 0:
                    ret = ret + str(weight_id)
            return ret

        for result in results:
            if result is None:
                continue
            teamInfo=result['args']['match_info'].get('teams',result['args']['match_info'])
            for team, info in teamInfo.items():
                policyName=info['Policy']#+info['Suffix']
                weight_id=info['Weight']
                res = np.mean([result['score'][p] for p in result['args']['player']
                    if result['policy_map'][p] == info['Policy']+info['Suffix']
                ])
                if(policyName==self.policy_to_train and weight_id<=0):
                    model_epoch = result['args']['model_epoch']
                    n, r, r2 = self.results.get(model_epoch, (0, 0, 0))
                    self.results[model_epoch] = n + 1, r + res, r2 + res ** 2
                    if model_epoch not in self.results_per_opponent:
                        self.results_per_opponent[model_epoch] = {}
                    opponent = ','.join(sorted([get_policy_label_for_log(op) for t,op in teamInfo.items() if t!=team]))
                    n, r, r2 = self.results_per_opponent[model_epoch].get(opponent, (0, 0, 0))
                    self.results_per_opponent[model_epoch][opponent] = n + 1, r + res, r2 + res ** 2
                self.summary_writer.add_scalar("evaluation/score/"+get_policy_label_for_log(info),res,self.num_returned_results)

            populate_config = ray.get(self.match_maker.onEpisodeEnd.remote(result['args']['match_info'],result['match_maker_result']))
            for policyName,conf in populate_config.items():
                self.populate(policyName, conf['weight_id'], conf['reset'])
            match_maker_metrics = ray.get(self.match_maker.get_metrics.remote(result['args']['match_info'],result['match_maker_result']))
            for k,v in match_maker_metrics.items():
                self.summary_writer.add_scalar("match_maker/"+k,v,self.num_returned_episodes+self.num_returned_results)
            self.num_returned_results += 1
        self.summary_writer.flush()

    def update(self):
        # call update to every component
        print("======= ",self.name," =======")
        print('epoch %d' % self.model_epoch)

        if self.model_epoch not in self.results:
            print('evaluation stats = Nan (0)')
        else:
            def output_wp(name, results):
                n, r, r2 = results
                mean = r / (n + 1e-6)
                name_tag = ' (%s)' % name if name != '' else ''
                std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
                print('evaluation stats%s = %.3f +- %.3f' % (name_tag, mean, std))

            keys = self.results_per_opponent[self.model_epoch]
            output_wp('total', self.results[self.model_epoch])
            for key in sorted(list(self.results_per_opponent[self.model_epoch])):
                output_wp(key, self.results_per_opponent[self.model_epoch][key])

        if self.model_epoch not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_epoch]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))

        model, steps = self.trainer.update()
        if model is None:
            model = self.models[self.policy_to_train]
        self.update_model(model, steps)
        self.summary_writer.add_scalar("trainer/num_returned_episodes",self.num_returned_episodes,self.model_epoch)
        self.summary_writer.add_scalar("trainer/num_returned_results",self.num_returned_results,self.model_epoch)
        self.summary_writer.add_scalar("trainer/num_returned_matches",self.num_returned_episodes+self.num_returned_results,self.model_epoch)

        # 経過時間による終了判定
        elapsedTime = time.time() - self.startedTime
        print(f'elapsedTime {elapsedTime :.1f} sec.')
        if 'time_limit' in self.args:
            if self.args['time_limit'] > 0 and elapsedTime > self.args['time_limit']:
                self.shutdown_flag = True
                print(f"Time limit reached.")

        # clear flags
        self.flags = set()
        print("=====================")

    def make_match(self, wid):
        if self.shutdown_flag:
            return None
        else:
            args = {'model_epoch': self.model_epoch}

            # decide role
            if self.num_results < self.eval_rate * self.num_episodes:
                args['role'] = 'e'
            else:
                args['role'] = 'g'
            args['match_type'] = (self.name+':'+args['role'], wid)
            args['match_info'] = ray.get(self.match_maker.makeNextMatch.remote(*args['match_type']))
            det_rates=self.args.get('deterministic',{'g':0.0,'e':0.0})
            teamInfo=args['match_info'].get('teams',args['match_info'])
            for team in teamInfo:
                policy_name = teamInfo[team]["Policy"]+teamInfo[team]["Suffix"]
                role = args['role'] if policy_name==self.policy_to_train else 'e'
                det_rate=det_rates.get(role,0.0)
                teamInfo[team]['deterministic'] = True if random.random() < det_rate else False
            if args['role'] == 'g':
                # genatation configuration
                self.num_episodes += 1

            elif args['role'] == 'e':
                # evaluation configuration
                self.num_results += 1

            return args

    def notifyWorkerShutdown(self,wid):
        self.workers.pop(wid)
        if(self.count_active_worker()==0 and self.shutdown_flag):
            self.summary_writer.close()
            print(self.name,'finished server')

    def count_active_worker(self):
        return sum([1 if w is not None else 0 for w in self.workers.values()])

    def isRunning(self):
        return self.count_active_worker() > 0 or not self.shutdown_flag

    def add_workers(self, number, reuse_index = True):
        if reuse_index:
            new_indices = [i+1 for i in range(len(self.workers.keys())+number) if i+1 not in self.workers]
        else:
            new_indices = [max(self.workers.keys())+2+i for i in range(number)]
        worker_options=self.args.get("worker_options",{})
        for i in new_indices:
            self.workers[i] = Worker.options(name=self.name+":Worker_"+str(i),**worker_options).remote(self.args,i)
            self.workers[i].run.remote()

    def removeDeadWorkers(self):
        indices = list(self.workers.keys())
        for i in indices:
            worker = self.workers[i]
            try:
                ray.get(worker.is_alive.remote())
            except ray.exceptions.RayActorError:
                self.notifyWorkerShutdown(i)
    def requestShutdown(self):
        self.shutdown_flag=True

def train_main(args):
    prepare_env(args['env_args'])  # preparing environment is needed in stand-alone mode
    #setup save directory
    save_dir_base = os.path.abspath(args.get('save_dir','.'))
    save_dir = os.path.join(
        save_dir_base,
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,'policies', 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'matches', 'checkpoints'), exist_ok=True)
    args['save_dir']=save_dir

    #setup MatchMaker
    match_maker_args = args.pop('match_maker_args')
    match_maker_args['weight_pool'] = os.path.join(save_dir, 'policies', 'weight_pool')
    match_maker_args['log_prefix'] = os.path.join(save_dir, 'matches', 'matches')
    match_maker_args['policy_config'] = args['policy_config']
    on_which_node=match_maker_args.get("node_designation",None)#MatchMakerを動かすノードのIPアドレスを指定
    if(on_which_node is None):
        #動作するノードを指定しなかった場合
        remoteMatchMakerClass=ray.remote(num_cpus=1)(args['custom_classes'][match_maker_args['match_maker_class']])
    else:
        #動作するノードを指定した場合
        if(on_which_node==args["ray_args"]["entrypoint_ip_address"]):
            on_which_node=args["ray_args"]["node_ip_address"] #127.0.0.1対策
        remoteMatchMakerClass=ray.remote(num_cpus=1,resources={"node:{}".format(on_which_node): 0.001})(args['custom_classes'][match_maker_args['match_maker_class']])
    match_maker = remoteMatchMakerClass.options(name='MatchMaker').remote(match_maker_args)

    #deploy Learners
    learners=[]
    train_args_all=args.pop('train_args')
    train_args_common=train_args_all.pop('common',{})
    for name, train_args in train_args_all.items():
        num_gpus=train_args.get("num_gpus",1)
        on_which_node=train_args.get("node_designation",None)#Learnerを動かすノードのIPアドレスを指定
        if(on_which_node is None):
            #動作するノードを指定しなかった場合
            remoteLearnerClass = ray.remote(num_gpus=num_gpus,num_cpus=1)(Learner)
        else:
            #動作するノードを指定した場合
            if(on_which_node==args["ray_args"]["entrypoint_ip_address"]):
                on_which_node=args["ray_args"]["node_ip_address"] #127.0.0.1対策
            remoteLearnerClass = ray.remote(num_gpus=num_gpus,num_cpus=1,resources={"node:{}".format(on_which_node): 0.001})(Learner)
        args['train_args']=copy.deepcopy(train_args_common)
        args['train_args'].update(train_args)
        args['name']=name
        learner=remoteLearnerClass.options(name=name,max_concurrency=2).remote(args=args)
        learners.append(learner)
    next(iter(learners)).setup_initial_models.remote()
    ray.get([l.run.remote() for l in learners])

    isRunning=True
    while isRunning:
        tmp=ray.get([l.isRunning.remote() for l in learners])
        for i,b in enumerate(tmp):
            if not b:
                learners[i]=None
        learners=[l for l in learners if l is not None]
        if len(learners)==0:
            isRunning=False
        time.sleep(10)
