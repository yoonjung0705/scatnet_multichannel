'''module that trains the LSTM and scat-transform + LSTM model given the simulated files'''
import os
import numpy as np
import scat_utils as scu
import sim_utils as siu
import net_utils as nu
import torch
from itertools import product

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/simulations/two_beads/'

'''common inputs'''
root_dir = ROOT_DIR
file_names_data = ['tbd_0.pt', 'tbd_1.pt']
#file_names_scat = []

n_epochs_max = 2000
train_ratio = 0.8
batch_size = 64
n_workers = 4

# RNN inputs
hidden_sizes = [20, 50, 100]
n_layerss = [2, 3]
bidirectionals = [True]

'''
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
                    nu.train_rnn(file_name_scat, [hidden_size, hidden_size], n_layers, bidirectional, classifier=False,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir)
                except:
                    print("exception occurred for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))


'''
# train RNNs for raw data
for file_name_data in file_names_data:
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_data, [hidden_size, hidden_size], n_layers, bidirectional, classifier=False,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir)
                except:
                    print("exception occurred for {}, hidden_size:{}, n_layers:{}, bidirectional:{}"\
                        .format(file_name_data, hidden_size, n_layers, bidirectional))

