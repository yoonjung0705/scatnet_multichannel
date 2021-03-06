'''module that processes and trains a classifier for the optical trap active bath experiment data'''
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
ROOT_DIR = './data/experiment/trap_bead_active_bath'
file_name_data = 'data.pt'
file_name_data_test = 'data_test.pt'

# common inputs
data_len = 2**11
root_dir = ROOT_DIR
# we take train_ratio amount of data which includes training and validation data
# within this data, we take train_ratio amount and use it for training.
# the remaining data is for test
# FIXME: change it so that you use train_ratio and val_ratio like below
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 1 - (train_ratio + val_ratio)
file_paths_data = glob.glob(os.path.join(root_dir, 'ad57_*.txt'))

# scat transform inputs
#avg_lens = [2**8]
avg_lens = [2**5, 2**7, 2**8, 2**9]
n_filter_octaves = [(1, 1)]
#n_filter_octaves = list(product([1,4], [1,4]))
# [(1,1), (1,2), (1,4), (2,1), (2,2), (2,4), (4,1), (4,2), (4,4)]
file_names_scat = []

# training inputs
n_epochs_max = 2000
batch_size = 100
n_workers = 4

# NN inputs
#n_nodes_hiddens = [] # FIXME: add later

# RNN inputs
hidden_sizes = [10, 50, 200, 500]
#hidden_sizes = [10, 50, 200, 500]
n_layerss = [2]
bidirectionals = [True]
lr = 0.001
betas = (0.9, 0.999)

file_data_lens = []
# determine number of timepoints per condition in case it differs among files
for file_path_data in file_paths_data:
    data = pd.read_csv(file_path_data, header=None, delimiter='\t')
    file_data_lens.append(len(data))

n_data = min(file_data_lens) // data_len
n_data_test = int(n_data * test_ratio)
n_data_train_val = n_data - n_data_test
print("Data length of the files:{}, taking {} timepoints for each file.".format(file_data_lens, n_data * data_len))
datas = []
labels = []
# load and split data
regex = r'ad57_([0-9]+p?[0-9]*)x_led_([0-9]+p?[0-9]*)_laser_150ma_[0-9]+.txt'

cs = [re.match(regex, os.path.basename(file_path_data)).group(1) for file_path_data in file_paths_data]
leds = [re.match(regex, os.path.basename(file_path_data)).group(2) for file_path_data in file_paths_data]
cs = np.array([float(c.replace('p', '.')) for c in cs])
leds = np.array([float(led.replace('p', '.')) for led in leds])

cs_uniq = np.unique(cs) # np.unique() also sorts the elements in ascending order
leds_uniq = np.unique(leds)
datas = []
datas_test = []
for c in cs_uniq:
    for led in leds_uniq:
        idxs_file = np.where((c == cs) & (led == leds))[0] # FIXME: multiple indices exist
        datas_tmp = []
        datas_tmp_test = []
        for idx_file in idxs_file:
            file_path_data = file_paths_data[idx_file]
            data_raw = pd.read_csv(file_path_data, sep='\t', header=None)
            idx_1 = n_data_train_val * data_len
            idx_2 = n_data * data_len
            data_tmp = (data_raw.iloc[:idx_1].loc[:, [0, 1]].values - data_raw.iloc[:idx_1].loc[:, [2, 3]].values).T # shaped (2, n_data_train_val * data_len)
            data_tmp = data_tmp.reshape([2, -1, data_len]).transpose([1, 0, 2]) # shaped (n_data_train_val, 2, data_len)
            datas_tmp.append(data_tmp)

            data_tmp_test = (data_raw.iloc[idx_1:idx_2].loc[:, [0, 1]].values - data_raw.iloc[idx_1:idx_2].loc[:, [2, 3]].values).T # shaped (2, n_data_test * data_len)
            data_tmp_test = data_tmp_test.reshape([2, -1, data_len]).transpose([1, 0, 2]) # shaped (n_data_test, 2, data_len)
            datas_tmp_test.append(data_tmp_test)

        data = np.concatenate(datas_tmp, axis=0) # shaped (n_trials * n_data_train_val, 2, data_len)
        data_test = np.concatenate(datas_tmp_test, axis=0) # shaped (n_trials * n_data_test, 2, data_len)

        datas.append(data)
        datas_test.append(data_test)

n_trials = len(idxs_file) # this should be same for every condition. For example, we should have 3 files per condition
data = np.stack(datas, axis=0).reshape([len(cs_uniq), len(leds_uniq), n_trials * n_data_train_val, 2, data_len])
data_test = np.stack(datas_test, axis=0).reshape([len(cs_uniq), len(leds_uniq), n_trials * n_data_test, 2, data_len])
samples = {'data':data, 'labels':[cs_uniq, leds_uniq], 'label_names':['c', 'led'], 'bacteria':'ad57', 'sample_rate':10000., 'laser_ma':150}
samples_test = {'data':data_test, 'labels':[cs_uniq, leds_uniq], 'label_names':['c', 'led'], 'bacteria':'ad57', 'sample_rate':10000., 'laser_ma':150}

torch.save(samples, os.path.join(root_dir, file_name_data))
torch.save(samples_test, os.path.join(root_dir, file_name_data_test))

# create scat transformed versions
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("scat transforming data_len:{} with parameters avg_len:{}, n_filter_octave:{}".format(data_len, avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform('data.pt', avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_name_test_scat = scu.scat_transform('data_test.pt', avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_names_scat.append(file_name_scat)
        except:
            print("exception for avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))

# train RNNs for scat transformed data
for file_name_scat in file_names_scat:
    meta = torch.load(os.path.join(root_dir, file_name_scat))
    avg_len = meta['avg_len']
    n_filter_octave = meta['n_filter_octave']
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_scat, hidden_size, n_layers, bidirectional, classifier=True,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)
                except:
                    print("exception for file_name_scat:{}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_scat, hidden_size, n_layers, bidirectional))

# train RNNs for raw data
for hidden_size in hidden_sizes:
    for n_layers in n_layerss:
        for bidirectional in bidirectionals:
            try:
                print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
                nu.train_rnn(file_name_data, hidden_size, n_layers, bidirectional, classifier=True,
                    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                    n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)
            except:
                print("exception for file_name_data:{}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))


