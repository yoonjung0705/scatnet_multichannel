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
train_ratio = 0.8 # used for both training and validation. rest is used for test
file_paths_data = glob.glob(os.path.join(root_dir, '*.h5'))

# scat transform inputs
avg_lens = [2**4, 2**6]
n_filter_octaves = [(1, 1)]

log_transform=False
filter_format='fourier_truncated'

track_lens = []
label_to_idx = {'nan':0, 'oo':1, 'mrlc':2}
labels = []
data_list = []
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
            data_list.append(track)
            idx_track_cum += 1

max_track_len = max(track_lens)
n_tracks_total = len(track_lens)

data = np.zeros((n_tracks_total, 2, max_track_len))
for idx_track_cum, track in enumerate(data_list):
    track_len = track_lens[idx_track_cum]
    data[idx_track_cum, :, :track_len] = track

labels = np.concatenate(labels, axis=0)
index = nu._train_test_split(n_tracks_total, train_ratio=train_ratio, seed=42)
data_train = data[index['train']]
data_test = data[index['test']]
labels_train = labels[index['train']]
labels_test = labels[index['test']]
track_lens_train = list(np.array(track_lens)[index['train']])
track_lens_test = list(np.array(track_lens)[index['test']])

samples_train = {'data':data_train, 'labels':labels_train, 'label_to_idx':label_to_idx, 'data_lens':track_lens_train}
samples_test = {'data':data_test, 'labels':labels_test, 'label_to_idx':label_to_idx, 'data_lens':track_lens_test}

file_path_train = os.path.join(root_dir, file_name_train)
file_path_test = os.path.join(root_dir, file_name_test)

torch.save(samples_train, file_path_train)
torch.save(samples_test, file_path_test)

# perform scat transform and append to list
file_names = ['data.pt', 'data_test.pt']
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        for file_name in file_names:
            print("scat transforming {} with parameters avg_len:{}, n_filter_octave:{}".format(file_name, avg_len, n_filter_octave))
            file_name_no_ext, _ = os.path.splitext(file_name)
            data_scat_list = []
            track_scat_lens = []
            file_path = os.path.join(root_dir, file_name)
            samples = torch.load(file_path)
            n_tracks = len(samples['data'])
            labels = samples['labels']
            for idx_track in range(n_tracks):
                track = samples['data'][idx_track]
                track_len = samples['data_lens'][idx_track]
                scat = scu.ScatNet(track_len, avg_len, n_filter_octave=n_filter_octave, filter_format=filter_format)
                S = scat.transform(track[np.newaxis, :, :track_len])
                S = scu.stack_scat(S)[0] # shaped (2, n_nodes, track_scat_len) where n_nodes is fixed but track_scat_len varies
                track_scat_len = S.shape[-1]
                track_scat_lens.append(track_scat_len)
                data_scat_list.append(S)

            max_track_scat_len = max(track_scat_lens)
            n_nodes = data_scat_list[0].shape[1]
            data_scat = np.zeros((n_tracks, 2, n_nodes, max_track_len))

            for idx_track in range(n_tracks):
                track_scat_len = track_scat_lens[idx_track]
                data_scat[idx_track, :, :, :track_scat_len] = data_scat_list[idx_track]

            nums = cu.match_filename(r'{}_scat_([0-9]+).pt'.format(file_name_no_ext), root_dir=root_dir)
            nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
            file_name_scat = '{}_scat_{}.pt'.format(file_name_no_ext, idx)
            file_path_scat = os.path.join(root_dir, file_name_scat)

            samples_scat = {'data':data_scat, 'labels':labels, 'label_to_idx':label_to_idx,
                    'avg_len':avg_len, 'log_transform':log_transform, 'n_filter_octave':n_filter_octave,
                    'filter_format':filter_format, 'file_name':file_name, 'data_lens':track_scat_lens}
            torch.save(samples_scat, file_path_scat)

'''
# check
idx = 1568
data = torch.load('data.pt')
data_scat = torch.load('data_scat_0.pt')
track_len = data['data_lens'][idx]
avg_len = data_scat['avg_len']
n_filter_octave = data_scat['n_filter_octave']
scat = scu.ScatNet(track_len, avg_len, n_filter_octave)
S = scat.transform(data['data'][idx][np.newaxis, :, :track_len])
S = scu.stack_scat(S)
track_scat_len = data_scat['data_lens'][idx]
data_scat['data'][idx][:,:,:track_scat_len] - S[0]
'''
