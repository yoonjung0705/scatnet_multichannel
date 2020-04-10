'''module that processes data prior to training'''
import os
import numpy as np
import pandas as pd
import re
import scat_utils as scu
import sim_utils as siu
import net_utils as nu
import torch
import glob
import h5py
from itertools import product
import scat_utils as scu

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/experiments/irfp'
file_name_train = 'data.pt'
file_name_test = 'data_test.pt'

# common inputs
root_dir = ROOT_DIR
# split data for training, validation, test
train_val_ratio = 0.8 # used for both training and validation. rest is used for test
file_paths_data = sorted(glob.glob(os.path.join(root_dir, '*.h5'))) # sort to make data consistent when seed given

# scat transform inputs
avg_lens = [2**4, 2**6]
n_filter_octaves = [(1, 1)]

log_transform=False
filter_format='fourier_truncated'

track_lens = []
labels_lut = [(0.,), (1.,), (2.,)] # label 0 means activity 0., etc.
labels = []
data = []
idx_track_cum = 0
for file_path_data in file_paths_data:
    file_name_data = os.path.basename(file_path_data)
    if file_name_data.startswith(('lonan', 'nan')):
        label = 0
    elif file_name_data.startswith('oo'):
        label = 1
    elif file_name_data.startswith('mrlc'):
        label = 2
    else:
        raise IOError("Invalid file name: should start with either lonan, nan, oo, or mrlc")

    with h5py.File(file_path_data, 'r') as f:
        n_tracks = len(f)
        labels.append(np.full((n_tracks,), fill_value=label))
        for idx_track in range(1, n_tracks + 1):
            track = f[str(idx_track)][()] # shaped (2, track_len)
            track_len = track.shape[1]
            track_lens.append(track_len)
            data.append(track)
            idx_track_cum += 1

n_tracks_total = len(track_lens)

labels = np.concatenate(labels, axis=0)
index = nu._train_test_split(n_tracks_total, train_ratio=train_val_ratio, seed=42)
data_train = [data[idx] for idx in index['train']]
data_test = [data[idx] for idx in index['test']]
#data_test = data[index['test']]
labels_train = labels[index['train']]
labels_test = labels[index['test']]
track_lens_train = list(np.array(track_lens)[index['train']])
track_lens_test = list(np.array(track_lens)[index['test']])

samples_train = {'data':data_train, 'labels':labels_train, 'label_names':['activity'], 'labels_lut':labels_lut}
samples_test = {'data':data_test, 'labels':labels_test, 'label_names':['activity'], 'labels_lut':labels_lut}

file_path_train = os.path.join(root_dir, file_name_train)
file_path_test = os.path.join(root_dir, file_name_test)

torch.save(samples_train, file_path_train)
torch.save(samples_test, file_path_test)

# perform scat transform and append to list
file_names = [file_name_train, file_name_test]
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        for file_name in file_names:
            print("scat transforming {} with parameters avg_len:{}, n_filter_octave:{}".format(file_name, avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform(file_name, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)

'''
# sanity check
idx = 168 
samples = torch.load('data.pt') 
samples_scat = torch.load('data_scat_0.pt') 
track_len = samples['data'][idx].shape[1] 
avg_len = samples_scat['avg_len'] 
n_filter_octave = samples_scat['n_filter_octave'] 
scat = scu.ScatNet(track_len, avg_len, n_filter_octave) 
S = scat.transform(samples['data'][idx][np.newaxis, :, :]) 
S = scu.stack_scat(S) 
samples_scat['data'][idx] - S[0] # should be all zero
'''
