'''module that computes the minimum value of the validation loss for given set of file
select a few files that show a low value of the validation loss, then plot them using plot_loss.py
for those that show reasonable training behavior, retrain to get the model and compute predictions on test data'''
import os
import torch
import glob
import re
import common_utils as cu
import scat_utils as scu

root_dir = './data/simulations/data_len_256_gamma_1_1p5/pos/models'
#root_dir = './data/simulations/data_len_256_gamma_1_1p5/disp/'
#root_dir = './data/experiments/irfp/models'
#root_dir = './data/experiments/irfp/'
#root_dir = './data/experiments/bead/2020_0305/data_len_256_poly/pos/'
# file_name_regexs elements must be enclosed with ()

# TWO BEADS
# POS
#file_name_regexs = ['(tbd_7_meta_rnn_[0-9]+_diff_coef_ratios.pt)', # raw model
#    '(tbd_7_scat_[0-9]+_meta_rnn_[0-9]+_diff_coef_ratios.pt)'] # scat model
file_name_regexs = ['(tbd_7_meta_rnn_[0-9]+_k_ratios.pt)', # raw model
    '(tbd_7_scat_[0-9]+_meta_rnn_[0-9]+_k_ratios.pt)'] # scat model

# DISP
#file_name_regexs = ['(tbd_4_disp_meta_rnn_[0-9]+_diff_coef_ratios.pt)', # raw model
#    '(tbd_4_disp_scat_[0-9]+_meta_rnn_[0-9]+_diff_coef_ratios.pt)'] # scat model
#file_name_regexs = ['(tbd_2_disp_meta_rnn_[0-9]+_k_ratios.pt)', # raw model
#    '(tbd_2_disp_scat_[0-9]+_meta_rnn_[0-9]+_k_ratios.pt)'] # scat model

# IRFP
#file_name_regexs = ['(data_meta_rnn_[0-9]+.pt)',
#    '(data_scat_[0-9]+_meta_rnn_[0-9]+.pt)',
#    '(data_disp_meta_rnn_[0-9]+.pt)',
#    '(data_disp_scat_[0-9]+_meta_rnn_[0-9]+.pt)']
    
#file_name_regexs = ['(data_meta_rnn_[0-9]+.pt)',
#    '(data_disp_meta_rnn_[0-9]+.pt)']

# BEAD
#file_name_regexs = ['(data_meta_rnn_[0-9]+.pt)',
#    '(data_scat_[0-9]+_meta_rnn_[0-9]+.pt)']
#file_name_regexs = ['(data_disp_meta_rnn_[0-9]+.pt)',
#    '(data_disp_scat_[0-9]+_meta_rnn_[0-9]+.pt)']
 
epoch_len_thresh = 30

file_names = []
for file_name_regex in file_name_regexs:
    file_names += cu.match_filename(file_name_regex, root_dir)
file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]

file_paths_tmp = []
for file_path in file_paths: 
    meta = torch.load(file_path, map_location='cpu') 
    if len(meta['epoch']) > epoch_len_thresh:
        file_paths_tmp.append(file_path)

file_paths = file_paths_tmp

n_files = len(file_paths)
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    meta = torch.load(file_path, map_location='cpu')

    loss = meta['loss']
    epoch = meta['epoch']
    if 'scat' in file_name:
        if 'disp' in file_name:
            #match = re.fullmatch('(tbd_[0-9]+_disp_scat_[0-9]+)_meta_rnn_[0-9]+_.*.pt', file_name) # two beads
            match = re.fullmatch('(data_disp_scat_[0-9]+)_meta_rnn_[0-9]+.pt', file_name) # irfp
        else:
            #match = re.fullmatch('(tbd_[0-9]+_scat_[0-9]+)_meta_rnn_[0-9]+_.*.pt', file_name) # two beads
            match = re.fullmatch('(data_scat_[0-9]+)_meta_rnn_[0-9]+.pt', file_name) # irfp
        #file_name_data = match.group(1) + '.pt'
        #samples = torch.load(os.path.join(root_dir, file_name_data))
        # ignore the first iteration's loss to better visualize the trend
        #print("file_name:{}\nhidden_size:{}\nn_layers:{}\navg_len:{}\nelapsed:{:.2f}\nloss_train_min:{:.5f}\nloss_val_min:{:.5f}\n".
        #    format(file_name, meta['hidden_size'], meta['n_layers'], samples['avg_len'], meta['elapsed'][-1],
        #    min(loss['train'][1:]), min(loss['val'][1:])))
        # ignore the first iteration's loss to better visualize the trend
        print("file_name:{}\nhidden_size:{}\nn_layers:{}\nelapsed:{:.2f}\nloss_train_min:{:.5f}\nloss_val_min:{:.5f}\n".
            format(file_name, meta['hidden_size'], meta['n_layers'], meta['elapsed'][-1],
            min(loss['train'][1:]), min(loss['val'][1:])))
    else:
        # ignore the first iteration's loss to better visualize the trend
        print("file_name:{}\nhidden_size:{}\nn_layers:{}\nelapsed:{:.2f}\nloss_train_min:{:.5f}\nloss_val_min:{:.5f}\n".
            format(file_name, meta['hidden_size'], meta['n_layers'], meta['elapsed'][-1],
            min(loss['train'][1:]), min(loss['val'][1:])))

