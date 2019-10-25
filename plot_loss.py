'''module that plots the training and validation loss as a function of number of epochs'''
import os
import torch
import matplotlib.pyplot as plt
import glob
import scat_utils as scu

#root_dir = './data/'
root_dir = '/home/yoonjung/SeagateSSHD/scat_data/trial_0'

# provide file names and paths manually
file_names = ['tbd_0_meta_rnn_0.pt' ,'tbd_1_meta_rnn_0.pt', 'tbd_2_meta_rnn_0.pt']
file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]

# provide file names and paths using regular expression
#file_paths = glob.glob(os.path.join(root_dir, 'tbd_*_scat_meta_rnn_0.pt'))
#file_names = [os.path.basename(file_path) for file_path in file_paths]

plt.style.use('dark_background')
fontsize_label = 14
fontsize_title = 18
fig_w = 12
fig_h = 8

plt.close('all')
n_files = len(file_names)

figs = []; axs = []
meta_0 = torch.load(file_paths[0])
n_labels = len(meta_0['label_names'])

for idx_file in range(n_files):
    file_path = file_paths[idx_file]
    file_name = file_names[idx_file]
    meta = torch.load(file_path)

    fig, ax = plt.subplots(1, n_labels, num=idx_file)
    fig.set_size_inches(fig_w * n_labels, fig_h)
    fig.suptitle(file_name)

    for idx_label in range(n_labels):
        loss = meta['loss_mean'][idx_label]
        epoch = meta['epoch'][idx_label]

        # ignore the first iteration's loss to better visualize the trend
        ax[idx_label].plot(epoch[1:], loss['train'][1:], label='train')
        ax[idx_label].plot(epoch[1:], loss['val'][1:], label='validation')
        ax[idx_label].set_title(meta['label_names'][idx_label].replace('_', ' '), fontsize=fontsize_title)
        ax[idx_label].set_xlabel('Epochs', fontsize=fontsize_label)
        ax[idx_label].set_ylabel('Average Loss', fontsize=fontsize_label)
        ax[idx_label].legend()
    
    figs.append(fig)
    axs.append(ax)


ylim_low = []
ylim_high = []

# match the axis limits
for idx_label in range(n_labels):
    ylim_low.append(min([ax[idx_label].get_ylim()[0] for ax in axs]))
    ylim_high.append(max([ax[idx_label].get_ylim()[1] for ax in axs]))

for idx_file in range(n_files):
    for idx_label in range(n_labels):
        axs[idx_file][idx_label].set_ylim(ylim_low[idx_label], ylim_high[idx_label])
        #axs[idx_file][idx_label].set_ylim(0, 40)
        figs[idx_file].suptitle(file_names[idx_file])

plt.show()
