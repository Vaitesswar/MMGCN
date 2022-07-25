from __future__ import print_function
import argparse
import os
import time
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
import easydict
import sys
sys.path.insert(0, './training')
from training import Processor

args = easydict.EasyDict({
    
    "model_saved_name": './runs/NTU_RGBD_60_mgn', # the folder for saved models
    "work_dir": './work_dir', # the work folder for storing results
    
    # processor
    "phase": 'train', # must be train or test
    "save_score": False, # if True, the classification score will be stored
    
    # visulize and debug
    "seed": 1, # 'random seed for pytorch'
    "log_interval": 100, # the interval for printing messages (#iteration)
    "save_interval": 1, # the interval for storing models (#iteration)
    "eval_interval": 5, # the interval for evaluating models (#iteration)
    "print_log": True, # print logging or not'
    "show_topk": [1, 5], # 'which Top K accuracy will be shown'
    
    # feeder
    "num_worker": 0, # the number of worker for data loader
    "train_feeder_args": {'joint_data_path': './data/NTU_RGBD_60/train_data_joint.npy',
                          'bone_data_path': './data/NTU_RGBD_60/train_data_bone.npy',
                          'joint_motion_data_path' : './data/NTU_RGBD_60/train_data_joint_motion.npy',
                          'label_path' : './data/NTU_RGBD_60/train_label.pkl',
                          'debug': False,
                          'random_choose': False,
                          'random_shift': False,
                          'random_move': False,
                          'window_size': -1,
                          'normalization': False}, # the arguments of data loader for training
    
    "test_feeder_args": {'joint_data_path': './data/NTU_RGBD_60/val_data_joint.npy',
                          'bone_data_path': './data/NTU_RGBD_60/val_data_bone.npy',
                          'joint_motion_data_path' : './data/NTU_RGBD_60/val_data_joint_motion.npy',
                          'label_path' : './data/NTU_RGBD_60/val_label.pkl',
                          'debug': False,
                          'random_choose': False,
                          'random_shift': False,
                          'random_move': False,
                          'window_size': -1,
                          'normalization': False}, # the arguments of data loader for test
    
    "weights": None, # the weights for network initialization
    "ignore_weights": [], # the name of weights which will be ignored in the initialization
    
    # optim
    "base_lr": 0.1, # initial learning rate
    "step": [60,90], # the epoch where optimizer reduce the learning rate
    "device": [0], # the indexes of GPUs for training or testing
    "optimizer": 'SGD', # type of optimizer
    "nesterov": True, # use nesterov or not
    "batch_size": 32, # training batch size
    "test_batch_size": 64, # test batch size
    "start_epoch": 0, # start training from which epoch
    "num_epoch": 120, # stop training in which epoch
    "weight_decay": 0.0005, # weight decay for optimizer
    "freeze_graph_until": 10 # number of epochs before making graphs learnable
})


processor = Processor(args)

processor.start()