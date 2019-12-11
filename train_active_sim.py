'''module that processes the active-passive particle simulation data and trains on it'''
import os
import re
import numpy as np
import pandas as pd
import scat_utils as scu
import sim_utils as siu
import net_utils as nu
import torch
import glob
from itertools import product

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/simulations/active_passive_sim'

# common inputs
data_len = 2**11
root_dir = ROOT_DIR
n_data = 300
n_data_test = 50
train_ratio = 0.8

# scat transform inputs
avg_lens = [2**4, 2**6, 2**8]
n_filter_octaves = list(product([1,4], [1,4]))
# [(1,1), (1,2), (1,4), (2,1), (2,2), (2,4), (4,1), (4,2), (4,4)]
file_names_scat = []
file_names_scat_test = []

# training inputs
n_epochs_max = 1000
batch_size = 100
n_workers = 4

# NN inputs
#n_nodes_hiddens = [] # FIXME: add later

# RNN inputs
hidden_sizes = [10, 50, 200, 500]
n_layerss = [2]
bidirectionals = [True]
lr = 0.001
betas = (0.9, 0.999)

dir_names = glob.glob(os.path.join(root_dir, 'Phia*'))
dir_names = [os.path.basename(dir_name) for dir_name in dir_names]
regex = r'Phia_([0-9p]+)_v_([0-9p]+)'
cs = [re.match(regex, dir_name).group(1) for dir_name in dir_names]
cs = np.array([float(c.replace('p', '.')) for c in cs])
vs = [re.match(regex, dir_name).group(2) for dir_name in dir_names]
vs = np.array([float(v.replace('p', '.')) for v in vs])

cs_uniq = np.unique(cs) # np.unique() also sorts the elements in ascending order
vs_uniq = np.unique(vs)
xys = []
xys_test = []
for c in cs_uniq:
    for v in vs_uniq:
        idx_dir_name = np.where((c == cs) & (v == vs))[0][0]
        dir_name = dir_names[idx_dir_name]
        data = pd.read_csv(os.path.join(root_dir, dir_name, 'pos'), sep='\t', skiprows=1, header=None).values
        x = np.reshape(data[:n_data * data_len, 0], [n_data, data_len])
        y = np.reshape(data[:n_data * data_len, 1], [n_data, data_len])
        xy = np.stack([x, y], axis=1) # shaped (n_data, 2, data_len)
        xys.append(xy)
        x_test = np.reshape(data[n_data * data_len:(n_data + n_data_test) * data_len, 0], [n_data_test, data_len])
        y_test = np.reshape(data[n_data * data_len:(n_data + n_data_test) * data_len, 1], [n_data_test, data_len])
        xy_test = np.stack([x_test, y_test], axis=1) # shaped (n_data, 2, data_len)
        xys_test.append(xy_test)
data = np.stack(xys, axis=0).reshape([len(cs_uniq), len(vs_uniq), n_data, 2, data_len])
data_test = np.stack(xys_test, axis=0).reshape([len(cs_uniq), len(vs_uniq), n_data_test, 2, data_len])
samples = {'data':data, 'labels':[cs_uniq, vs_uniq], 'label_names':['c', 'v'], 'Dr':1, 'Dt':0.05, 'T':25, 'K':10, 'dt':0.001}
samples_test = {'data':data_test, 'labels':[cs_uniq, vs_uniq], 'label_names':['c', 'v'], 'Dr':1, 'Dt':0.05, 'T':25, 'K':10, 'dt':0.001}

nums = cu.match_filename(r'pos_([0-9]+).pt', root_dir=root_dir)
nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
file_name_data = 'pos_{}.pt'.format(idx)
file_name_data_test = 'pos_{}.pt'.format(idx + 1)

torch.save(samples, os.path.join(root_dir, file_name_data))
torch.save(samples_test, os.path.join(root_dir, file_name_data_test))

# create scat transformed versions
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("scat transforming data_len:{} with parameters avg_len:{}, n_filter_octave:{}".format(data_len, avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_name_scat_test = scu.scat_transform(file_name_data_test, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_names_scat.append(file_name_scat)
            file_names_scat_test.append(file_name_scat_test)
        except:
            pass

# train RNNs for scat transformed data
for file_name_scat in file_names_scat:
    meta = torch.load(os.path.join(root_dir, file_name_scat))
    avg_len = meta['avg_len']
    n_filter_octave = meta['n_filter_octave']
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"\
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_scat, [hidden_size, hidden_size], n_layers, bidirectional,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)
                except:
                    pass


# train RNNs for raw data
for hidden_size in hidden_sizes:
    for n_layers in n_layerss:
        for bidirectional in bidirectionals:
            try:
                print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
                nu.train_rnn(file_name_data, [hidden_size, hidden_size], n_layers, bidirectional,
                    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                    n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)
            except:
                pass
