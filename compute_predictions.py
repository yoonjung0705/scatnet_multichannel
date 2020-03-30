'''this is a module that loads the meta data of the neural net and computes the performance'''
# FIXME: only manually (roughly) checked for rnn scat case for both experiment data and simulation data.
# Have not checked rnn raw data case.

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

min_loss_epochs = [8000, 4000] # max 39 for log every 50. max 199 for log every 10.
#root_dir = './data/simulations/data_len_256_gamma_1_1p5/model_candidates'
#root_dir = './data/experiments/bead/2020_0228/'
root_dir = './data/experiments/bead/2020_0305/data_len_256_poly_train_val_ratio_0p2/models'
#root_dir = './data/experiments/irfp'

# file name of test data
# TWO BEADS
#file_names_test = ['tbd_0_test.pt',
#        'tbd_0_test_scat_1.pt',
#        'tbd_0_test.pt',
#        'tbd_0_test_scat_1.pt']
#file_name_test = 'tbd_0_test_scat_0.pt'
#file_name_test = 'tbd_0_test_scat_1.pt'

# IRFP
file_names_test = ['data_test.pt', 'data_test_scat_0.pt']


# LIST of file names of trained models
# TWO BEADS
#file_names_meta = ['tbd_0_meta_rnn_21_diff_coef_ratios.pt',
#        'tbd_0_scat_1_meta_rnn_15_diff_coef_ratios.pt',
#        'tbd_0_meta_rnn_22_k_ratios.pt',
#        'tbd_0_scat_1_meta_rnn_9_k_ratios.pt']


#file_names_meta = ['tbd_9_meta_rnn_9_diff_coef_ratios.pt']
#file_names_meta = ['tbd_9_scat_0_meta_rnn_9_k_ratios.pt']
#file_names_meta = ['tbd_9_scat_0_meta_rnn_9_diff_coef_ratios.pt']
#file_names_meta = ['tbd_9_scat_1_meta_rnn_9_k_ratios.pt']
#file_names_meta = ['tbd_9_scat_1_meta_rnn_9_diff_coef_ratios.pt']

# IRFP
file_names_meta = ['data_meta_rnn_11.pt', 'data_scat_0_meta_rnn_11.pt']
# OR, provide file names and paths using regular expression
#file_paths_meta = glob.glob(os.path.join(root_dir, 'tbd_0_scat_meta_rnn_*.pt'))
#file_names_meta = [os.path.basename(file_path) for file_path in file_paths]
#file_names = ['tbd_1_scat.pt'] * len(file_paths_meta)

batch_size = 10 # batch size when performing forward propagation on test data using trained weights

file_paths_test = [os.path.join(root_dir, file_name_test) for file_name_test in file_names_test]
file_paths_meta = [os.path.join(root_dir, file_name_meta) for file_name_meta in file_names_meta]
n_files = len(file_paths_meta)

for idx_file in range(n_files):

    file_path_meta = file_paths_meta[idx_file]
    file_name_meta = file_names_meta[idx_file]
    file_path_test = file_paths_test[idx_file]

    samples = torch.load(file_path_test)
    data, labels, label_names = samples['data'], samples['labels'], samples['label_names']
    n_data_total = len(data) 

    meta = torch.load(file_path_meta)
    # compute the nearest epoch number
    min_loss_epoch = min_loss_epochs[idx_file]
    idx_min_loss_epoch = np.argmin(np.abs((np.array(meta['epoch']) - min_loss_epoch)))
    classifier = meta['classifier']
    if not meta['classifier']:
        idx_label = samples['label_names'].index(meta['label_names'])
        labels = labels[idx_label]
        label_names = label_names[idx_label]
    # reshape data. output is shaped (n_data_total, n_channels * (n_scat_nodes), data_len).
    # (n_scat_nodes) means 1 if data not transformed
    if isinstance(data, np.ndarray):
        data = np.reshape(data, (n_data_total, -1, data.shape[-1]))
    elif isinstance(data, list):
        data = [np.reshape(data_slice, (-1, data_slice.shape[-1])) for data_slice in data]
    else:
        raise ValueError("Invalid type of data given")

    input_size = data[0].shape[0]
    output_size = meta['output_size']
    dataset = nu.TimeSeriesDataset(data, labels, transform=nu.ToTensor())
    dataloader = DataLoader(dataset, sampler=SequentialSampler(range(n_data_total)),
        batch_size=batch_size, collate_fn=nu.collate_fn)

    rnn = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'],
        output_size=meta['output_size'], n_layers=meta['n_layers'], bidirectional=meta['bidirectional']).cuda()
    rnn.load_state_dict(meta['model'][idx_min_loss_epoch])

    #criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    #metric = 'cross_entropy_mean' if classifier else 'rmse'
    loss_sum = {}
    loss_metric = {}
    loss_sum = 0.
    outputs = []
    for batch in dataloader:
        # permute s.t. shape is (data_len, n_data_total, n_channels * (n_scat_nodes))
        batch_data = batch['data'].permute([2, 0, 1]).cuda()
        #batch_labels = batch['labels']
        input_lens = batch['input_lens'].type(torch.cuda.LongTensor)
        output = rnn(batch_data, input_lens=input_lens)
        # for regression, output of rnn is shaped (batch_size, 1). drop dummy axis
        if classifier:
            output = output.argmax(axis=1)
        else:
            output = output[:, 0]
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.detach().cpu().numpy()
    if classifier:
        accuracy = sum(outputs == np.array(labels)) / n_data_total
        print("file_name:{}, accuracy:{}".format(file_name_meta, accuracy))
    else:
        rmse = np.sqrt(sum((outputs - labels)**2) / n_data_total)
        print("file_name:{}, rmse:{}".format(file_name_meta, rmse))


