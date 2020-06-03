'''module that plots the training and validation loss as a function of number of epochs for one or multiple trained networks'''
import os
import torch
import matplotlib.pyplot as plt
import glob

import common_utils as cu
import scat_utils as scu

#plt.style.use('dark_background')
fontsize_label = 14
fontsize_title = 18
fig_w = 12
fig_h = 8

#root_dir = './data/simulations/data_len_128_gamma_1_3/tbd_3'
root_dir = './data/simulations_for_junang/data_len_2048_gamma_1_3_k_1_7_t_4_10/models'
#root_dir = './data/experiments/bead/2020_0319/data_len_1024_train_val_26_test_26_sep/models'
#root_dir = './data/experiments/bead/2020_0319/data_len_512_train_val_177_test_59/models'
#root_dir = './data/experiments/irfp/models'
save_fig = True
"""
# file_name_regexs elements must be enclosed with ()
file_name_regexs = ['(tbd_0_scat_[0-9]+_meta_rnn_[0-9]+_diff_coef_ratios.pt)',
    '(tbd_0_meta_rnn_[0-9]+_diff_coef_ratios.pt)'] 
file_names = []
for file_name_regex in file_name_regexs:
    file_names += cu.match_filename(file_name_regex, root_dir)
file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]

"""

epoch_thresh = 1950 # only consider files that have at least this number of epochs processed

#file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]
file_paths = glob.glob(os.path.join(root_dir, '*.pt'))

file_paths_tmp = []
plt.close('all')
for file_path in file_paths: 
    meta = torch.load(file_path, map_location='cpu') 
    if 'epoch' in meta.keys():
        if meta['epoch']:
            if max(meta['epoch']) >= epoch_thresh:
                file_paths_tmp.append(file_path)
    del meta

file_paths = file_paths_tmp

n_files = len(file_paths)

figs = []; axs = [];
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    meta = torch.load(file_path, map_location='cpu')

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_w, fig_h)
    #fig.suptitle(file_name)

    loss = meta['loss']
    epoch = meta['epoch']

    # ignore the first iteration's loss to better visualize the trend
    ax.plot(epoch[1:], loss['train'][1:], label='train')
    ax.plot(epoch[1:], loss['val'][1:], label='validation')
    ax.set_title("{}, elapsed:{:.2f}, loss_train_min:{:.5f}, loss_val_min:{:.5f}".
        format(meta['label_names'], meta['elapsed'][-1],
        min(loss['train'][1:]), min(loss['val'][1:])), fontsize=fontsize_title)
    ax.set_xlabel('Epochs', fontsize=fontsize_label)
    ax.set_ylabel(meta['criterion'], fontsize=fontsize_label)
    ax.legend()
    
    figs.append(fig)
    axs.append(ax)

# match the axis limits
ylim_low = min([ax.get_ylim()[0] for ax in axs])
ylim_high = max([ax.get_ylim()[1] for ax in axs])

for idx_file in range(n_files):
    file_path = file_paths[idx_file]
    file_name = os.path.basename(file_path)
    axs[idx_file].set_ylim(ylim_low, ylim_high)
    figs[idx_file].suptitle(file_name)

    if save_fig:
        file_name_no_ext, _ = os.path.splitext(file_name)
        file_name_fig = file_name_no_ext + '.png'
        file_path_fig = os.path.join(root_dir, file_name_fig)
        figs[idx_file].savefig(file_path_fig)

plt.show()

