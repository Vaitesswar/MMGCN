import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import json
import random
import math
from tqdm import tqdm

sys.path.insert(0, '../')
import tools

class Feeder(Dataset):
    def __init__(self, joint_data_path, bone_data_path,joint_motion_data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        
        self.label_path = label_path
        try:
            with open(label_path, 'rb') as f:
                self.file_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(label_path, 'rb') as f:
                self.file_name, self.label = pickle.load(f, encoding='latin')
        self.time_steps = 52

    def __len__(self):
        if 'train' in self.label_path:
            return len(self.label)*5
        else:
            return len(self.label)
                   

    def __iter__(self):
        return self
    
    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        file_name, label = self.file_name[index % len(self.label)], self.label[index % len(self.label)]
        with open('./data/ucla/' + file_name, 'r') as f:
            json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons) # T,V,C
        view = int(file_name[-7:-5])
        if view != 3: # training (views 1 and 2)
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0,1,:]
            value = value - center

            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            joint_data = np.zeros( (self.time_steps, 20, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length))*100, self.time_steps)
            random_idx.sort()
            joint_data[:,:,:] = value[random_idx,:,:]

        else: # test (view 3)
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0,1,:]
            value = value - center

            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1

            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            joint_data = np.zeros( (self.time_steps, 20, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            idx = np.linspace(0,length-1,self.time_steps).astype(np.int)
            joint_data[:,:,:] = value[idx,:,:] # T,V,C
            
        joint_data = np.transpose(joint_data, (2, 0, 1))
        C,T,V = joint_data.shape
        joint_data = np.reshape(joint_data,(C,T,V,1))
        
        bone = [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3),
            (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), 
            (11, 10),(12, 11), (13, 1), (14, 13), (15, 14),
            (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
        bone_data = np.zeros_like(joint_data)
        for bone_idx in range(20):
            bone_data[:, :, bone[bone_idx][0] - 1, :] = joint_data[:, :, bone[bone_idx][0] - 1, :] - joint_data[:, :, bone[bone_idx][1] - 1, :]
        
        motion_data = np.zeros_like(joint_data)
        motion_data[:, :-1, :, :] = joint_data[:, 1:, :, :] - joint_data[:, :-1, :, :]
        
        # Either label is fine
        return joint_data, motion_data, bone_data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort() # Sorts in an ascending manner (The default in numpy is -1 (the last axis))
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)] # Checks if label matches any top_k predictions 
        return sum(hit_top_k) * 1.0 / len(hit_top_k)