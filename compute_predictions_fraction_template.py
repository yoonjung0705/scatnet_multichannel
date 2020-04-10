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

device = 'cuda:0' # or cuda:0

# for using part of the test data to make error bars, set the start and end index for the data
idx_start = <IDX_START>
idx_end = idx_start + 100

#min_loss_epochs = [1200, 1000, 9500, 7500, 350, 400, 2300, 2300] # 512 k
#min_loss_epochs = [2000, 6000, 3500, 8500, 350, 1000, 3500, 3500] # 512 diff coef

#min_loss_epochs = [2500, 1500, 3000, 3500, 500, 500, 700, 700] # 1024 k
#min_loss_epochs = [2700, 1200, 3000, 9500, 400, 550, 550, 1300] # 1024 diff coef

#min_loss_epochs = [1000, 5000, 7000, 9000, 200, 250, 1000, 1500] # 2048 k
min_loss_epochs = [2000, 5000, 9000, 5000, 300, 300, 1000, 3000] # 2048 diff coef


#min_loss_epochs = [5500, 700] # 512_train_val_81_test_702
#min_loss_epochs = [10000, 1700] # 512_train_val_405_test_378
#min_loss_epochs = [4600, 2500] # 2048_train_val_81_test_108
#min_loss_epochs = [9300, 1800] # 2048_train_val_135_test_54

root_dir = './data/simulations/data_len_2048_gamma_1_3_k_1_7_t_4_10/models'
#root_dir = './data/experiments/bead/2020_0305/data_len_2048_train_val_135_test_54/models'
#root_dir = './data/experiments/bead/2020_0228/'
#root_dir = './data/experiments/bead/2020_0305/data_len_256_poly_train_val_ratio_0p2/models'
#root_dir = './data/experiments/irfp'

# file name of test data
# TWO BEADS
#file_names_test = ['tbd_0_test.pt',
#        'tbd_0_test_scat_1.pt',
#        'tbd_0_test.pt',
#        'tbd_0_test_scat_1.pt']
#file_name_test = 'tbd_0_test_scat_0.pt'
#file_name_test = 'tbd_0_test_scat_1.pt'

file_names_test = [
    'tbd_0_test.pt',
    'tbd_0_test.pt',
    'tbd_0_test.pt',
    'tbd_0_test.pt',
    'tbd_0_test_scat_0.pt',
    'tbd_0_test_scat_0.pt',
    'tbd_0_test_scat_0.pt',
    'tbd_0_test_scat_0.pt',
    ]

#file_names_test = ['data_test.pt', 'data_test_scat_0.pt']

# IRFP



# LIST of file names of trained models
# TWO BEADS
"""
file_names_meta = [
    'tbd_0_meta_rnn_4_k_ratios.pt',
    'tbd_1_meta_rnn_5_k_ratios.pt',
    'tbd_2_meta_rnn_0_k_ratios.pt',
    'tbd_3_meta_rnn_1_k_ratios.pt',
    'tbd_0_scat_0_meta_rnn_3_k_ratios.pt',
    'tbd_1_scat_0_meta_rnn_1_k_ratios.pt',
    'tbd_2_scat_0_meta_rnn_1_k_ratios.pt',
    'tbd_3_scat_0_meta_rnn_0_k_ratios.pt',
    ]
file_names_meta = [
    'tbd_0_meta_rnn_2_diff_coef_ratios.pt',
    'tbd_1_meta_rnn_4_diff_coef_ratios.pt',
    'tbd_2_meta_rnn_1_diff_coef_ratios.pt',
    'tbd_3_meta_rnn_0_diff_coef_ratios.pt',
    'tbd_0_scat_0_meta_rnn_2_diff_coef_ratios.pt',
    'tbd_1_scat_0_meta_rnn_0_diff_coef_ratios.pt',
    'tbd_2_scat_0_meta_rnn_0_diff_coef_ratios.pt',
    'tbd_3_scat_0_meta_rnn_1_diff_coef_ratios.pt',
    ]
"""


"""
file_names_meta = [
    cu.match_filename('(tbd_0_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_1_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_2_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_3_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_0_scat_0_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_1_scat_0_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_2_scat_0_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    cu.match_filename('(tbd_3_scat_0_meta_rnn_[0-9]+_k_ratios.pt)', root_dir),
    ]
for file_name_meta in file_names_meta:
    assert(len(file_name_meta) == 1), "Invalid number of files. Should be only 1 trained model for each case"
file_names_meta = [file_name_meta[0] for file_name_meta in file_names_meta]

"""
file_names_meta = [
    cu.match_filename('(tbd_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_1_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_2_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_3_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_0_scat_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_1_scat_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_2_scat_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    cu.match_filename('(tbd_3_scat_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)', root_dir),
    ]
for file_name_meta in file_names_meta:
    assert(len(file_name_meta) == 1), "Invalid number of files. Should be only 1 trained model for each case"
file_names_meta = [file_name_meta[0] for file_name_meta in file_names_meta]




#file_names_meta = ['data_meta_rnn_1.pt', 'data_scat_0_meta_rnn_1.pt']

# IRFP
#file_names_meta = ['data_meta_rnn_11.pt', 'data_scat_0_meta_rnn_11.pt']
# OR, provide file names and paths using regular expression
#file_paths_meta = glob.glob(os.path.join(root_dir, 'tbd_0_scat_meta_rnn_*.pt'))
#file_names_meta = [os.path.basename(file_path) for file_path in file_paths]
#file_names = ['tbd_1_scat.pt'] * len(file_paths_meta)

batch_size = 40 # batch size when performing forward propagation on test data using trained weights

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
    if device == 'cpu':
        meta = torch.load(file_path_meta, map_location='cpu')
    else:
        meta = torch.load(file_path_meta, map_location='cuda:0')
    # compute the nearest epoch number
    min_loss_epoch = min_loss_epochs[idx_file]
    idx_min_loss_epoch = np.argmin(np.abs((np.array(meta['epoch']) - min_loss_epoch)))
    classifier = meta['classifier']
    elapsed = meta['elapsed'][idx_min_loss_epoch]
    epoch = meta['epoch'][idx_min_loss_epoch]
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

    # take out only a fraction of the test data
    data = data[idx_start:idx_end]
    labels = labels[idx_start:idx_end]
    n_data_total = len(data)

    input_size = data[0].shape[0]
    output_size = meta['output_size']
    dataset = nu.TimeSeriesDataset(data, labels, transform=nu.ToTensor())
    dataloader = DataLoader(dataset, sampler=SequentialSampler(range(n_data_total)),
        batch_size=batch_size, collate_fn=nu.collate_fn, num_workers=0)

    if device == 'cpu':
        rnn = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'],
            output_size=meta['output_size'], n_layers=meta['n_layers'], bidirectional=meta['bidirectional'])
    else:
        rnn = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'],
            output_size=meta['output_size'], n_layers=meta['n_layers'], bidirectional=meta['bidirectional']).cuda()
    rnn.load_state_dict(meta['model'][idx_min_loss_epoch])
    del meta
    #criterion = nn.CrossEntropyLoss(reduction='sum') if classifier else nn.MSELoss(reduction='sum')
    #metric = 'cross_entropy_mean' if classifier else 'rmse'
    loss_sum = {}
    loss_metric = {}
    loss_sum = 0.
    outputs = []
    for batch in dataloader:
        # permute s.t. shape is (data_len, n_data_total, n_channels * (n_scat_nodes))
        if device == 'cpu':
            batch_data = batch['data'].permute([2, 0, 1])
            input_lens = batch['input_lens'].type(torch.LongTensor)
        else:
            batch_data = batch['data'].permute([2, 0, 1]).cuda()
            input_lens = batch['input_lens'].type(torch.cuda.LongTensor)
        output = rnn(batch_data, input_lens=input_lens)
        # for regression, output of rnn is shaped (batch_size, 1). drop dummy axis
        if classifier:
            output = output.argmax(axis=1).detach().cpu().numpy()
        else:
            output = output[:, 0].detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs, axis=0)
    if classifier:
        accuracy = sum(outputs == np.array(labels)) / n_data_total
        print("file_name:{}, epoch:{}, accuracy(%):{:.2f}, elapsed:{:.1f}"
            .format(file_name_meta, epoch, accuracy * 100, elapsed))
    else:
        rmse = np.sqrt(sum((outputs - labels)**2) / n_data_total)
        print("file_name:{}, epoch:{}, rmse:{:.2f}, elapsed:{:.1f}"
            .format(file_name_meta, epoch, rmse, elapsed))


