'''module that processes and trains the optical trap passive experiment data'''
import os
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
ROOT_DIR = './data/experiment/position_stiffness'

# common inputs
data_len = 2**11
root_dir = ROOT_DIR
# we take train_ratio amount of data which includes training and validation data
# within this data, we take train_ratio amount and use it for training.
# the remaining data is for test
train_ratio = 0.8

file_names_data = glob.glob(os.path.join(root_dir, 'L_*.csv'))


# scat transform inputs
avg_lens = [2**5, 2**7, 2**9]
n_filter_octaves = list(product([1,4], [1,4]))
# [(1,1), (1,2), (1,4), (2,1), (2,2), (2,4), (4,1), (4,2), (4,4)]
file_names_scat = []

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
lr = 0.0001
betas = (0.9, 0.999)

file_data_lens = []
# determine number of timepoints per condition in case it differs among files
for file_name_data in file_names_data:
    data = pd.read_csv(file_name_data, header=None)
    file_data_lens.append(len(data))

data_len_total = min(file_data_lens) - 1 # the first element is the stiffness
print("Data length of the files:{}, taking {} timepoints for each file.".format(file_data_lens, data_len_total))
k = data.loc[0, 0]
data = data.loc[1:, 0].values
datas = []
labels = []
# load and split data
for file_name_data in file_names_data:
    data = pd.read_csv(file_name_data, header=None)
    k = data.loc[0, 0]
    data = data.loc[1:data_len_total + 1, 0].values
    n_data = data_len_total // data_len
    data_len_total = n_data * data_len
    data = data[:data_len_total].reshape([-1, 1, data_len]) # shaped [n_data, 1, data_len]
    datas.append(data)
    labels.append(k)

idx = nu._train_test_split(n_data, train_ratio=train_ratio)
data = np.stack(datas, axis=0) # shaped (n_conditions, n_data, 1, data_len)
samples_train = {'data':data[:, idx['train'], :, :], 'label_names':['k'], 'labels':[np.array(labels)]}
samples_test = {'data':data[:, idx['test'], :, :], 'label_names':['k'], 'labels':[np.array(labels)]}
        
torch.save(samples_train, os.path.join(root_dir, 'obd_exp_0.pt'))
torch.save(samples_test, os.path.join(root_dir, 'obd_exp_1.pt'))

# create scat transformed versions
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("scat transforming data_len:{} with parameters avg_len:{}, n_filter_octave:{}".format(data_len, avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform('obd_exp_0.pt', avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_name_test_scat = scu.scat_transform('obd_exp_1.pt', avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_names_scat.append(file_name_scat)
        except:
            print("exception occurred during scat transformation for data_len:{} with parameters avg_len:{}, n_filter_octave:{}".format(data_len, avg_len, n_filter_octave))

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
                    nu.train_rnn(file_name_scat, [hidden_size], n_layers, bidirectional, classifer=False,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)
                except:
                    print("exception occurred during training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
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
                        n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)
                except:
                    print("exception occurred during training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}"\
                        .format(file_name_data, hidden_size, n_layers, bidirectional))
'''

