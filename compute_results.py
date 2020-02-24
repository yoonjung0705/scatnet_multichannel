'''this is a module that loads the meta data of the neural net (FC and/or RNN) and computes the performance'''
# FIXME: only manually (roughly) checked for rnn scat case for both experiment data and simulation data.
# Have not checked rnn raw data case. Have not checked any nn case at all.

'''standard imports'''
import os
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.pyplot import cm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

'''custom libraries'''
import common_utils as cu
import scat_utils as scu
import net_utils as nu

#root_dir = './data/simulations/'
root_dir = './data/experiment/trap_bead_active_bath'
root_dir_results = os.path.join(root_dir, 'results')

# provide file names (TEST data!) and paths manually
#file_names = ['tbd_1_scat_0.pt']
#file_names_meta = ['tbd_0_scat_0_meta_rnn_0.pt']

#file_names = ['data_test_scat_0.pt']
file_names = ['data_test.pt']
#file_names_meta = ['data_scat_0_meta_rnn_0.pt', 'data_scat_0_meta_rnn_1.pt', 'data_scat_0_meta_rnn_2.pt', 'data_scat_0_meta_rnn_3.pt']
file_names_meta = ['data_meta_rnn_2.pt']
file_names = file_names * len(file_names_meta)

# OR, provide file names and paths using regular expression
#file_paths_meta = glob.glob(os.path.join(root_dir, 'tbd_0_scat_meta_rnn_*.pt'))
#file_names_meta = [os.path.basename(file_path) for file_path in file_paths]
#file_names = ['tbd_1_scat.pt'] * len(file_paths_meta)

batch_size = 100 # batch size when performing forward propagation on test data using trained weights

# add .pt extension if not provided
file_names = [os.path.splitext(file_name)[0] + '.pt' for file_name in file_names]
file_names_meta = [os.path.splitext(file_name_meta)[0] + '.pt' for file_name_meta in file_names_meta]

file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]
file_paths_meta = [os.path.join(root_dir, file_name_meta) for file_name_meta in file_names_meta]
n_files = len(file_paths)

# make file_names into list
if isinstance(file_names, str):
    file_names = [file_names]

'''sanity check'''
# check number of files match with number of meta files
assert(len(file_names) == len(file_names_meta))

# if the training was done on scat transformed data, it should be tested on scat transformed data, too.
# if the training was done on pure time series data, it should be tested on pure time series data, too.
file_names_meta_is_scat = ['scat' in file_name_meta for file_name_meta in file_names_meta]
file_names_is_scat = ['scat' in file_name for file_name in file_names]
assert(file_names_meta_is_scat == file_names_is_scat), "The training and test data are not transformed in the same manner"

"""
# FIXME: these commented assertions should only run for simulated two beads data or poisson, one bead, diffusion data. Not for experimental data

# check if all meta files are trained on the same dataset
file_names_meta_common = [re.fullmatch(r'([a-z]{3}_[0-9]+)_?.*_meta.*', file_name_meta).group(1) for file_name_meta in file_names_meta]
assert(len(set(file_names_meta_common)) == 1), "The given neural networks are trained on different data sets"

# check if all meta files are testing on the same dataset
file_names_common = [re.fullmatch(r'([a-z]{3}_[0-9]+).*', file_name).group(1) for file_name in file_names]
assert(len(set(file_names_common)) == 1), "The given neural networks are tested on different data sets"

# training and testing should NOT be done on same data.
# since the source of data for files in file_names_meta are all identical and
# the source of data for files in file_names are all identical, we only need to compare the first element of file_names_meta_common and
# first element of file_names_common
assert(file_names_meta_common[0] != file_names_common[0]), "Training and testing cannot be done on same data"
"""

for idx_file in range(n_files):
    file_name = file_names[idx_file]
    file_path = file_paths[idx_file]
    file_path_meta = file_paths_meta[idx_file]
    file_name_meta = file_names_meta[idx_file]

    samples = torch.load(file_path)
    data = samples['data']
    transformed = 'scat' in file_name

    meta = torch.load(file_path_meta, map_location='cpu')

    if meta['classifier']:
        labels = samples['labels'] # list of arrays
        labels = np.array(list(product(*samples['labels'])), dtype='float32')
        n_conditions = len(labels)
        n_samples_total = data.shape[2] # number of data per condition
        if 'meta_rnn' in file_path_meta:
            if transformed:
                n_data_total = np.prod(data.shape[:-3])
                n_features = np.prod(data.shape[-3:-1])
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features, data_len])
            else:
                n_data_total = np.prod(data.shape[:-2])
                n_features = data.shape[-2]
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features, data_len])
            
            input = torch.tensor(input, dtype=torch.get_default_dtype())
            net = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'], output_size=n_conditions, n_layers=meta['n_layers'], bidirectional=meta['bidirectional'])
            net.load_state_dict(meta['weights'])

            dataset = nu.TimeSeriesDataset(input, np.zeros((n_data_total,)))
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(n_data_total)))
            output = []
            for idx_batch, batch in enumerate(dataloader):
                output.append(net(batch['data'].permute([2, 0, 1])).detach().numpy())
        else: # nn
            if transformed:
                n_data_total = np.prod(data.shape[:-3])
                n_features = np.prod(data.shape[-3:-1])
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features * data_len])
            else:
                n_data_total = np.prod(data.shape[:-2])
                n_features = data.shape[-2]
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features * data_len])

            input = torch.tensor(input, dtype=torch.get_default_dtype())
            net = nu.Net(meta['n_nodes'])
            net.load_state_dict(meta['weights'])

            dataset = nu.TimeSeriesDataset(input, np.zeros((n_data_total,)))
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(n_data_total)))
            output = []
            for idx_batch, batch in enumerate(dataloader):
                output.append(net(batch['data']).detach().numpy())

        output = np.concatenate(output, axis=0)
        output = output.argmax(axis=1)
        labels = np.arange(n_conditions).repeat(n_samples_total) # shaped (n_conditions * n_samples_total,)
        accuracy = (output == labels).mean()

        result = {'labels':samples['labels'], 'label_names':samples['label_names'],
                'prediction':output, 'file_name':file_name, 'root_dir':root_dir, 'accuracy':accuracy}
        file_name_test = os.path.splitext(file_name_meta)[0] + '_test_on_' + os.path.splitext(file_name)[0] + '.pt'
        file_path_test = os.path.join(root_dir_results, file_name_test)
        torch.save(result, file_path_test)

    else: # if regressor
        n_labels = len(meta['label_names'])
        labels = samples['labels'] # shaped (n_conditions, n_labels)
        rmses = []
        outputs = []
        if 'meta_rnn' in file_path_meta:
            if transformed:
                n_data_total = np.prod(data.shape[:-3])
                n_features = np.prod(data.shape[-3:-1])
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features, data_len])
            else:
                n_data_total = np.prod(data.shape[:-2])
                n_features = data.shape[-2]
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features, data_len])
            
            input = torch.tensor(input, dtype=torch.get_default_dtype())
            for idx_label in range(n_labels):
                net = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'][idx_label], output_size=1, n_layers=meta['n_layers'], bidirectional=meta['bidirectional'])
                net.load_state_dict(meta['weights'][idx_label])

                dataset = nu.TimeSeriesDataset(input, np.zeros((n_data_total,)))
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(n_data_total)))
                output = []
                for idx_batch, batch in enumerate(dataloader):
                    output.append(net(batch['data'].permute([2, 0, 1])).detach().numpy()[:, 0])
                output = np.concatenate(output, axis=0)
                outputs.append(output)
                mse = np.sum((output - labels[:, idx_label])**2) / n_data_total
                rmse = np.sqrt(mse)
                rmses.append(rmse)
            outputs = np.stack(outputs, axis=0)

        else: # nn
            if transformed:
                n_data_total = np.prod(data.shape[:-3])
                n_features = np.prod(data.shape[-3:-1])
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features * data_len])
            else:
                n_data_total = np.prod(data.shape[:-2])
                n_features = data.shape[-2]
                data_len = data.shape[-1]
                input = data.reshape([n_data_total, n_features * data_len])

            input = torch.tensor(input, dtype=torch.get_default_dtype())
            for idx_label in range(n_labels):
                net = nu.Net(meta['n_nodes'])
                net.load_state_dict(meta['weights'][idx_label])

                dataset = nu.TimeSeriesDataset(input, np.zeros((n_data_total,)))
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(n_data_total)))
                output = []
                for idx_batch, batch in enumerate(dataloader):
                    output.append(net(batch['data']).detach().numpy()[:, 0])
                output = np.concatenate(output, axis=0)
                outputs.append(output)
                mse = np.sum((output - labels[:, idx_label])**2) / n_data_total
                rmse = np.sqrt(mse)
                rmses.append(rmse)
            outputs = np.stack(outputs, axis=0)

        result = {'labels':samples['labels'], 'label_names':samples['label_names'],
                'prediction':outputs, 'file_name':file_name, 'root_dir':root_dir, 'rmse':rmses}
        file_name_test = os.path.splitext(file_name_meta)[0] + '_test_on_' + os.path.splitext(file_name)[0] + '.pt'
        file_path_test = os.path.join(root_dir_results, file_name_test)
        torch.save(result, file_path_test)

