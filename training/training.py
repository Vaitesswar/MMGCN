from __future__ import print_function
import argparse
import os
import time
import math
import numpy as np
import yaml
import pickle
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import MultiStepLR
import random
import inspect
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import easydict
import sys
sys.path.insert(0, './model')
from directed_multiple_temp import Model # Modify this line to change model architecture
sys.path.insert(0, './feeders')
from feeder import Feeder  # Modify this line to change feeder

class Processor():
    """Processor for Skeleton-based Action Recgnition"""
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()     

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
            
        self.global_step = 0
        self.load_model()
        self.load_param_groups()    # Group parameters to apply different learning rules
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.nan_count = 0

    def load_data(self):
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                worker_init_fn=torch.manual_seed(1),
                pin_memory=True)

        # Load test data regardless
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            worker_init_fn=torch.manual_seed(1),
            pin_memory=True)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        # Copy model file to output dir
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model().cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # Load weights
        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # Parallelise data if mulitple GPUs
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)


    def load_optimizer(self):
        p_groups = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                p_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                p_groups,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = '[ {} ] {}'.format(localtime, s)
        print(s)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def load_param_groups(self):
        self.param_groups = defaultdict(list)
        for name, params in self.model.named_parameters():
            if ('adj_mat' in name):
                self.param_groups['graph'].append(params)
            else:
                self.param_groups['other'].append(params)

        # NOTE: Different parameter groups should have different learning behaviour
        self.optim_param_groups = {
            'graph': {'params': self.param_groups['graph']},
            'other': {'params': self.param_groups['other']}
        }

    def update_graph_freeze(self, epoch):
        graph_requires_grad = (epoch > self.arg.freeze_graph_until)
        self.print_log('Graphs are {} at epoch {}'.format('learnable' if graph_requires_grad else 'frozen', epoch + 1))
        for param in self.param_groups['graph']:
            param.requires_grad = graph_requires_grad
        # graph_weight_decay = 0 if freeze_graphs else self.arg.weight_decay
        # NOTE: will decide later whether we need to change weight decay as well
        # self.optim_param_groups['graph']['weight_decay'] = graph_weight_decay

    def train(self, epoch, save_model=False):
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        self.update_graph_freeze(epoch)

        process = tqdm(loader)
        for batch_idx, (joint_data, joint_motion_data, bone_data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                joint_data = joint_data.float().cuda(self.output_device,non_blocking=True)
                joint_motion_data = joint_motion_data.float().cuda(self.output_device,non_blocking=True)
                bone_data = bone_data.float().cuda(self.output_device,non_blocking=True)
                label = label.long().cuda(self.output_device,non_blocking=True)
            timer['dataloader'] += self.split_time()

            # Clear gradients
            self.optimizer.zero_grad()

            joint_data.requires_grad_()
            joint_motion_data.requires_grad_()
            bone_data.requires_grad_()
                                
            # forward
            output = self.model(joint_data, joint_motion_data, bone_data)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0

            loss = self.loss(output, label)


            loss.backward()
            loss_values.append(loss.detach().item())
            timer['model'] += self.split_time()

                # Display loss
            process.set_description('loss: {:.4f}'.format(loss.detach().item()))

            value, predict_label = torch.max(output, 1)
            acc = torch.mean((predict_label == label).float())

            # Step after looping over batch splits
            self.optimizer.step()

            # Statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

        # Statistics of time consumption and loss
        proportion = {
            k: '{: 2d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_values)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        self.lr_scheduler.step(epoch)

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            loss_values, score_batches = [], []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (joint_data, joint_motion_data, bone_data, label, index) in enumerate(process):
                step += 1
                with torch.no_grad():
                    joint_data = joint_data.float().cuda(self.output_device,non_blocking=True)
                    joint_motion_data = joint_motion_data.float().cuda(self.output_device,non_blocking=True)
                    bone_data = bone_data.float().cuda(self.output_device,non_blocking=True)
                    label = label.long().cuda(self.output_device,non_blocking=True)
                    output = self.model(joint_data, joint_motion_data, bone_data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0

                    loss = self.loss(output, label)
                    score_batches.append(output.cpu().numpy())
                    loss_values.append(loss.item())
                    # Argmax over logits = labels
                    _, predict_label = torch.max(output, dim=1)

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.cpu().numpy())
                        for i, pred in enumerate(predict):
                            if result_file is not None:
                                f_r.write('{},{}\n'.format(pred, true[i]))
                            if pred != true[i] and wrong_file is not None:
                                f_w.write('{},{},{}\n'.format(index[i], pred, true[i]))

            # Concatenate along the batch dimension, and 1st dim ~= `len(dataset)`
            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' Model: ', self.arg.model_saved_name)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_values)))

            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            print('Best accuracy: {}, epoch: {}, model_name: {}'
                  .format(self.best_acc, self.best_acc_epoch, self.arg.model_saved_name))

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
