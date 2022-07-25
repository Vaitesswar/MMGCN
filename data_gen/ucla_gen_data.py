import os
import json
import numpy as np
import random
import math
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    file_list = sorted(os.listdir('./data/ucla')) # storage folder of json files
    time_steps = 52
    num_joint = 20
    max_body_true = 1
    channels = 3
    train_data = 1020
    val_data = 464
    train_label = list()
    train_filename = list()
    val_label = list()
    val_filename = list()
    corrected_labels = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 8:7, 9:8, 11:9, 12:10}

    # Joint data
    for i in tqdm(range(len(file_list))):
        file_name = file_list[i]
        view = int(file_name[-7:-5])
        if view != 3: # training (views 1 and 2)
            label = int(corrected_labels[int(file_name[1:3])])-1
            train_label.append(label) # Storing action classes
            train_filename.append(file_name) # Storing file names

        else: # test (view 3)
            label = int(corrected_labels[int(file_name[1:3])])-1
            val_label.append(label) # Storing action classes
            val_filename.append(file_name) # Storing file names

    with open('{}/{}_label.pkl'.format(out_path, 'train'), 'wb') as f:
            pickle.dump((train_filename, list(train_label)), f)
    with open('{}/{}_label.pkl'.format(out_path, 'val'), 'wb') as f:
            pickle.dump((val_filename, list(val_label)), f)