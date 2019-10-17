'''module that plots the training and validation loss as a function of number of epochs'''
import os
import torch
import matplotlib.pyplot as plt

import scat_utils as scu

root_dir = './data/'
file_name = 'tbd_0_scat_meta_rnn_2.pt'

fontsize_label = 14
fontsize_title = 18
fig_w = 12
fig_h = 8

file_path = os.path.join(root_dir, file_name)
meta = torch.load(file_path)

n_labels = len(meta['label_names'])
fig, ax = plt.subplots(1, n_labels)
fig.set_size_inches(fig_w * n_labels, fig_h)
for idx in range(n_labels):
    loss = meta['loss_mean'][idx]
    epoch = meta['epoch'][idx]

    # ignore the first iteration's loss to better visualize the trend
    ax[idx].plot(epoch[1:], loss['train'][1:], label='train')
    ax[idx].plot(epoch[1:], loss['val'][1:], label='validation')
    ax[idx].set_title(meta['label_names'][idx].replace('_', ' '), fontsize=fontsize_title)
    ax[idx].set_xlabel('Epochs', fontsize=fontsize_label)
    ax[idx].set_ylabel('Average Loss', fontsize=fontsize_label)

plt.show()
