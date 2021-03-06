'''module that processes experimental bacterial bath without optical trap data prior to training. mbd means multiple beads'''
# NOTE: this script might use only a fraction of the data for training which results in fewer number of classes
# NOTE: this script might mix the train+validation and test data to randomize the split

import os
import numpy as np
import pandas as pd
import re
import scat_utils as scu
import sim_utils as siu
import net_utils as nu
import torch
import glob
from itertools import product

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/experiments/bead/2020_0319'
file_name_data = 'data.pt'
file_name_data_test = 'data_test.pt'

# common inputs
data_len = 2**8 # timepoints is 2500 per condition. max is 1024 for this script's implementation.
root_dir = ROOT_DIR
# we take train_val_ratio amount of data which includes training and validation data
# within this data, we take train_ratio amount which is set in net_utils.py's train_rnn() and use it for training.
# in other words, we only determine the amount of train+val amount here. How it's divided between train, val is 
# determined when you actually do the training.
# the remaining data is for test
train_val_ratio = 0.75
test_ratio = 1 - (train_val_ratio)
file_paths_data = glob.glob(os.path.join(root_dir, 'ad57_*.txt'))
samples = {'label_names':['cs', 'lasers'], 'bacteria':'ad57', 'sample_rate_hz':None} # FIXME: add this info
samples_test = {'label_names':['cs', 'lasers'], 'bacteria':'ad57', 'sample_rate_hz':None}

# scat transform inputs
avg_lens = [2**6]
n_filter_octaves = [(1, 1)]

file_data_lens = []
# determine number of timepoints per track in case it differs among files
for file_path_data in file_paths_data:
    data = pd.read_csv(file_path_data, header=None, delimiter='\t').loc[:, [0, 1]]
    file_data_lens.append(len(data))

n_data = min(file_data_lens) // data_len
n_data_test = int(n_data * test_ratio)
n_data_train_val = n_data - n_data_test
print("Shortest trajectory length:{}".format(min(file_data_lens)))
print("Longest trajectory length:{}".format(max(file_data_lens)))
print("n_data per trajectory (total):{}".format(n_data))
print("n_data per trajectory (training + validation):{}".format(n_data_train_val))
print("n_data per trajectory (test):{}".format(n_data_test))

datas = []
labels = []
# load and split data
regex = r'ad57_([0-9]+p?[0-9]*)x_([0-9]+)um_laser_([0-9]+)_([0-9]+).txt'

cs = [re.match(regex, os.path.basename(file_path_data)).group(1) for file_path_data in file_paths_data]
d_um = set([re.match(regex, os.path.basename(file_path_data)).group(2) for file_path_data in file_paths_data])
lasers = [re.match(regex, os.path.basename(file_path_data)).group(3) for file_path_data in file_paths_data]
trial = [re.match(regex, os.path.basename(file_path_data)).group(4) for file_path_data in file_paths_data]
cs = np.array([float(c.replace('p', '.')) for c in cs])
lasers = np.array([float(laser) for laser in lasers])

cs_uniq = np.unique(cs) # np.unique() also sorts the elements in ascending order
lasers_uniq = np.unique(lasers)

# use only a fraction of the data
cs_uniq = cs_uniq[[0, 2]]
lasers_uniq = lasers_uniq[[0, 2]]

labels_lut = [(c, laser) for c in cs_uniq for laser in lasers_uniq]

datas = []
datas_test = []
labels = []
labels_test = []
label = 0

for c in cs_uniq:
    for laser in lasers_uniq:
        idxs_file = np.where((c == cs) & (laser == lasers))[0] # output of np.where() is a tuple, so we do [0]
        # NOTE: idx_file is a list of file indices. same condition but different beads
        datas_tmp = []
        datas_tmp_test = []
        for idx_file in idxs_file:
            file_path_data = file_paths_data[idx_file]
            data_raw = pd.read_csv(file_path_data, sep='\t', header=None).loc[:, [0, 1]]
            idx_1 = n_data_train_val * data_len
            idx_2 = n_data * data_len
            data_tmp = (data_raw.iloc[:idx_1].loc[:, [0, 1]].values).T # shaped (2, n_data_train_val * data_len)
            data_tmp = data_tmp.reshape([2, -1, data_len]).transpose([1, 0, 2]) # shaped (n_data_train_val, 2, data_len)
            datas_tmp.append(data_tmp)

            data_tmp_test = (data_raw.iloc[idx_1:idx_2].loc[:, [0, 1]].values).T # shaped (2, n_data_test * data_len)
            data_tmp_test = data_tmp_test.reshape([2, -1, data_len]).transpose([1, 0, 2]) # shaped (n_data_test, 2, data_len)
            datas_tmp_test.append(data_tmp_test)

            labels += [label] * n_data_train_val
            labels_test += [label] * n_data_test

        data = np.concatenate(datas_tmp, axis=0) # shaped (n_trials * n_data_train_val, 2, data_len)
        data_test = np.concatenate(datas_tmp_test, axis=0) # shaped (n_trials * n_data_test, 2, data_len)

        datas.append(data)
        datas_test.append(data_test)
        label += 1

#n_trials = len(idxs_file) # this should be same for every condition. For example, we should have 3 files (3 beads) per condition
#data = np.stack(datas, axis=0).reshape([len(cs_uniq) * len(lasers_uniq) * n_trials * n_data_train_val, 2, data_len])
data = np.concatenate(datas, axis=0)
#data_test = np.stack(datas_test, axis=0).reshape([len(cs_uniq) * len(lasers_uniq) * n_trials * n_data_test, 2, data_len])
data_test = np.concatenate(datas_test, axis=0)
#labels = np.array(range(len(labels_lut)), dtype=int).repeat(n_trials * n_data_train_val).tolist()
#labels_test = np.array(range(len(labels_lut)), dtype=int).repeat(n_trials * n_data_test).tolist()

# mix the data (so far the data was split sequentially into train+validation vs test)
# the following further randomly mixes trajectories to split into train+validation vs test
data_total = np.concatenate([data, data_test], axis=0)
n_data_total = data_total.shape[0]
index = nu._train_test_split(n_data_total, train_val_ratio, seed=42)
data = data_total[index['train']]
data_test = data_total[index['test']]

labels_total = labels + labels_test
labels = list(np.array(labels_total)[index['train']])
labels_test = list(np.array(labels_total)[index['test']])


samples.update({'data':data, 'labels':labels, 'd_um':d_um, 'labels_lut':labels_lut})
samples_test.update({'data':data_test, 'labels':labels_test, 'd_um':d_um, 'labels_lut':labels_lut})

torch.save(samples, os.path.join(root_dir, file_name_data))
torch.save(samples_test, os.path.join(root_dir, file_name_data_test))

# create scat transformed versions
file_names = [file_name_data, file_name_data_test]
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        for file_name in file_names:
            print("scat transforming {} with parameters avg_len:{}, n_filter_octave:{}".format(file_name, avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform(file_name, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)

