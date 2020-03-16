'''module that processes experimental bacterial bath + optical trap data prior to training'''
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
ROOT_DIR = './data/experiments/bead/2020_0305'
#ROOT_DIR = './data/experiments/bead/2020_0228'
file_name_data = 'data.pt'
file_name_data_test = 'data_test.pt'

# common inputs
data_len = 2**8 # timepoints is ~15000 per condition. don't set this larger than 2**9
root_dir = ROOT_DIR
# we take train_ratio amount of data which includes training and validation data
# within this data, we take train_ratio amount and use it for training.
# the remaining data is for test
train_ratio = 0.6
val_ratio = 0.1
test_ratio = 1 - (train_ratio + val_ratio)
file_paths_data = glob.glob(os.path.join(root_dir, 'ad57_*.txt')) # polydisperse
#file_paths_data = glob.glob(os.path.join(root_dir, 'ad57_*_5um_*.txt')) # monodisperse
samples = {'label_names':['cs', 'leds'], 'bacteria':'ad57', 'sample_rate_hz':50.}
samples_test = {'label_names':['cs', 'leds'], 'bacteria':'ad57', 'sample_rate_hz':50.}

# scat transform inputs
avg_lens = [2**3, 2**5]
n_filter_octaves = [(1, 1)]

file_data_lens = []
# determine number of timepoints per condition in case it differs among files
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
regex = r'ad57_([0-9]+p?[0-9]*)x_([0-9]+)um_led_([0-9]+p?[0-9]*)_laser_([0-9]+)ma.txt'

cs = [re.match(regex, os.path.basename(file_path_data)).group(1) for file_path_data in file_paths_data]
d_um = set([re.match(regex, os.path.basename(file_path_data)).group(2) for file_path_data in file_paths_data])
leds = [re.match(regex, os.path.basename(file_path_data)).group(3) for file_path_data in file_paths_data]
laser_ma = [re.match(regex, os.path.basename(file_path_data)).group(4) for file_path_data in file_paths_data]
cs = np.array([float(c.replace('p', '.')) for c in cs])
leds = np.array([float(led.replace('p', '.')) for led in leds])

assert(len(np.unique(laser_ma))), "Invalid data given. Laser power should be identical for all files"
laser_ma = float(laser_ma[0])

cs_uniq = np.unique(cs) # np.unique() also sorts the elements in ascending order
leds_uniq = np.unique(leds)
labels_lut = [(c, led) for c in cs_uniq for led in leds_uniq]

datas = []
datas_test = []
for c in cs_uniq:
    for led in leds_uniq:
        idxs_file = np.where((c == cs) & (led == leds))[0] # output of np.where() is a tuple, so we do [0]
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

        data = np.concatenate(datas_tmp, axis=0) # shaped (n_trials * n_data_train_val, 2, data_len)
        data_test = np.concatenate(datas_tmp_test, axis=0) # shaped (n_trials * n_data_test, 2, data_len)

        datas.append(data)
        datas_test.append(data_test)

n_trials = len(idxs_file) # this should be same for every condition. For example, we should have 3 files (3 beads) per condition
data = np.stack(datas, axis=0).reshape([len(cs_uniq) * len(leds_uniq) * n_trials * n_data_train_val, 2, data_len])
data_test = np.stack(datas_test, axis=0).reshape([len(cs_uniq) * len(leds_uniq) * n_trials * n_data_test, 2, data_len])
labels = np.array(range(len(labels_lut)), dtype=int).repeat(n_trials * n_data_train_val).tolist()
labels_test = np.array(range(len(labels_lut)), dtype=int).repeat(n_trials * n_data_test).tolist()
samples.update({'data':data, 'labels':labels, 'laser_ma':laser_ma, 'd_um':d_um, 'labels_lut':labels_lut})
samples_test.update({'data':data_test, 'labels':labels_test, 'laser_ma':laser_ma, 'd_um':d_um, 'labels_lut':labels_lut})

torch.save(samples, os.path.join(root_dir, file_name_data))
torch.save(samples_test, os.path.join(root_dir, file_name_data_test))

# create scat transformed versions
file_names = ['data.pt', 'data_test.pt']
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        for file_name in file_names:
            print("scat transforming {} with parameters avg_len:{}, n_filter_octave:{}".format(file_name, avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform(file_name, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)

