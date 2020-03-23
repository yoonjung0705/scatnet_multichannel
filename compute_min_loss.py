'''module that computes the minimum value of the validation loss for given set of file
select a few files that show a low value of the validation loss, then plot them using plot_loss.py
for those that show reasonable training behavior, retrain to get the model and compute predictions on test data'''
import os
import torch
import glob

import common_utils as cu
import scat_utils as scu

root_dir = './data/simulations/data_len_256_gamma_1_1p5/pos/'
# file_name_regexs elements must be enclosed with ()
file_name_regexs = ['(tbd_0_scat_[0-9]+_meta_rnn_[0-9]+_diff_coef_ratios.pt)',
    '(tbd_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)'] 
epoch_len = 200

file_names = []
for file_name_regex in file_name_regexs:
    file_names += cu.match_filename(file_name_regex, root_dir)
file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]

file_paths_tmp = []
for file_path in file_paths: 
    meta = torch.load(file_path) 
    if len(meta['epoch']) == epoch_len:
        file_paths_tmp.append(file_path)

file_paths = file_paths_tmp

n_files = len(file_paths)
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    meta = torch.load(file_path)

    loss = meta['loss']
    epoch = meta['epoch']

    # ignore the first iteration's loss to better visualize the trend
    print("file_name:{}\nhidden_size:{}\nn_layers:{}\nelapsed:{:.2f}\nloss_train_min:{:.5f}\nloss_val_min:{:.5f}".
        format(file_name, meta['hidden_size'], meta['n_layers'], meta['elapsed'][-1],
        min(loss['train'][1:]), min(loss['val'][1:])))
