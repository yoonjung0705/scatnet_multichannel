'''this is a module that loads the meta data of the neural net (FC and/or RNN) and generates figures that compare/show the performance'''
# FIXME: for now we test prediction results for data simulated with hyperparameters that were used for the training data.
# Later, include data simulated with hyperparameters whose values were not the ones used for the training data

# FIXME: output results are 1 for k, 2 for diff ratios for all trajectories when tested for rnn. This is also for scat transformed trajectories
# FIXME: check if there are ..meta..._test.pt files and if so, add number 0, 1, 2,... just like the meta file naming convention

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
import scat_utils as scu
import net_utils as nu

file_names = ['tbd_2.pt', 'tbd_2_scat.pt']
#file_names_meta = ['tbd_0_meta_rnn_0.pt', 'tbd_0_scat_meta_rnn_0.pt']
file_names_meta = ['tbd_0_meta_nn_0.pt', 'tbd_0_scat_meta_nn_0.pt']
root_dir = './data/'
epochs = [[50, 50], [50, 50]]

# plt.style.use('dark_background')
fontsize_title = 18
fontsize_label = 14
batch_size = 1000 # batch size when performing forward propagation on test data using trained weights

'''sanity check'''
# add .pt extension if not provided
file_names = [os.path.splitext(file_name)[0] + '.pt' for file_name in file_names]
file_names_meta = [os.path.splitext(file_name_meta)[0] + '.pt' for file_name_meta in file_names_meta]

file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]
file_paths_meta = [os.path.join(root_dir, file_name_meta) for file_name_meta in file_names_meta]
n_files = len(file_paths)

# make file_names into list
if isinstance(file_names, str):
    file_names = [file_names]

# check number of files match with number of meta files
assert(len(file_names) == len(file_names_meta))

# check if all meta files are trained on the same dataset
file_names_meta_common = [re.fullmatch(r'([a-z]{3}_[0-9]+)_?.*_meta.*', file_name_meta).group(1) for file_name_meta in file_names_meta]
assert(len(set(file_names_meta_common)) == 1), "The given neural networks are trained on different data sets"

# check if all meta files are testing on the same dataset
file_names_common = [re.fullmatch(r'([a-z]{3}_[0-9]+).*', file_name).group(1) for file_name in file_names]
assert(len(set(file_names_common)) == 1), "The given neural networks are tested on different data sets"

# if the training was done on scat transformed data, it should be tested on scat transformed data, too.
# if the training was done on pure time series data, it should be tested on pure time series data, too.
file_names_meta_is_scat = ['scat' in file_name_meta for file_name_meta in file_names_meta]
file_names_is_scat = ['scat' in file_name for file_name in file_names]
assert(file_names_meta_is_scat == file_names_is_scat), "The training and test data are not transformed in the same manner"

for idx_file in range(n_files):
    #fig, ax = plt.subplots()
    file_name = file_names[idx_file]
    file_path = file_paths[idx_file]
    file_path_meta = file_paths_meta[idx_file]
    file_name_meta = file_names_meta[idx_file]

    samples = torch.load(file_path)
    data = samples['data']
    meta = torch.load(file_path_meta, map_location='cpu')

    n_labels = len(meta['label_names'])
    epochs_file = epochs[idx_file]
    idx_epochs = [epoch_file // 10 for epoch_file in epochs_file]
    transformed = 'scat' in file_path
    outputs = []
    output_means = []
    output_stds = []
    labels_targets = []
    if 'meta_rnn' in file_path_meta:
        if transformed:
            n_data_total = np.prod(data.shape[:-3])
            n_features = np.prod(data.shape[-3:-1])
            data_len = data.shape[-1]
            input = data.reshape(data, [n_data_total, n_features, data_len])
        else:
            n_data_total = np.prod(data.shape[:-2])
            n_features = data.shape[-2]
            data_len = data.shape[-1]
            input = data.reshape([n_data_total, n_features, data_len])
        
        input = torch.tensor(input, dtype=torch.get_default_dtype())
        # following is shaped (n_labels, n_conditions)
        labels = np.array(list(product(*samples['labels'])), dtype='float32').swapaxes(0, 1)
        for idx_label in range(n_labels):
            net = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'], output_size=1, n_layers=meta['n_layers'], bidirectional=meta['bidirectional'])
            net.load_state_dict(meta['weights'][idx_label][idx_epochs[idx_label]])

            dataset = nu.TimeSeriesDataset(input, np.zeros((n_data_total,)))
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(n_data_total)))
            output = []
            print("forward propagation for {}".format(meta['label_names'][idx_label]))
            for idx_batch, batch in enumerate(dataloader):
                output.append(net(batch['data'].permute([2, 0, 1])).detach().numpy()[:, 0])
                print("{} out of {} samples propagated".format(min((idx_batch + 1) * batch_size, n_data_total), n_data_total))
            output = np.concatenate(output, axis=0)

            if transformed:
                output = output.reshape(samples['data'].shape[:-3])
                output_mean = output.mean(axis=-1)
                output_std = output.std(axis=-1)
            else:
                output = output.reshape(samples['data'].shape[:-2])
                output_mean = output.mean(axis=-1)
                output_std = output.std(axis=-1)
            labels_target = labels[idx_label].reshape(output_mean.shape)
            outputs.append(output)
            output_means.append(output_mean)
            output_stds.append(output_std)
            labels_targets.append(labels_target)

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
        # following is shaped (n_labels, n_conditions)
        labels = np.array(list(product(*samples['labels'])), dtype='float32').swapaxes(0, 1)
        for idx_label in range(n_labels):
            net = nu.Net(meta['n_nodes'])
            net.load_state_dict(meta['weights'][idx_label][idx_epochs[idx_label]])

            dataset = nu.TimeSeriesDataset(input, np.zeros((n_data_total,)))
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(n_data_total)))
            output = []
            print("forward propagation for {}".format(meta['label_names'][idx_label]))
            for idx_batch, batch in enumerate(dataloader):
                output.append(net(batch['data']).detach().numpy()[:, 0])
                print("{} out of {} samples propagated".format(min((idx_batch + 1) * batch_size, n_data_total), n_data_total))
            output = np.concatenate(output, axis=0)

            if transformed:
                output = output.reshape(samples['data'].shape[:-3])
                output_mean = output.mean(axis=-1)
                output_std = output.std(axis=-1)
            else:
                output = output.reshape(samples['data'].shape[:-2])
                output_mean = output.mean(axis=-1)
                output_std = output.std(axis=-1)
            labels_target = labels[idx_label].reshape(output_mean.shape)
            outputs.append(output)
            output_means.append(output_mean)
            output_stds.append(output_std)
            labels_targets.append(labels_target)

    result = {'labels':samples['labels'], 'label_names':samples['label_names'], 'prediction':outputs, 'epochs':epochs_file, 'file_name':file_name, 'root_dir':root_dir}
    torch.save(result, os.path.join(root_dir, os.path.splitext(file_name_meta)[0] + '_test.pt'))
            
    if n_labels == 1:
        fig, ax = plt.subplots()
        ax.errorbar(labels_target, output_mean, yerr=output_std, ls='', capsize=10, fmt='o')
        ax.errorbar(labels_target, labels_target, ls='--')
        ax.set_title(label_names[0].replace('_', ' ').capitalize(), fontsize=fontsize_title)
        ax.set_xlabel('Ground truth', fontsize=fontsize_label)
        ax.set_ylabel('Prediction', fontsize=fontsize_label)
        plt.show()
    elif n_labels == 2:
        n_conditions = len(labels_target)
        color = cm.rainbow(np.linspace(0, 1, n_conditions))
        fig, ax = plt.subplots()
        for i in range(n_conditions):
            ax.errorbar(output_means[0][i], output_means[1][i], xerr=output_stds[0][i], yerr=output_stds[1][i], fmt='o', capsize=10, c=color[i])
            ax.plot([output_means[0][i], labels_targets[0][i]], [output_means[1][i], labels_targets[1][i]], marker='o', color=color[i])

        ax.set_xlabel(samples['label_names'][0], fontsize=fontsize_label)
        ax.set_ylabel(samples['label_names'][1], fontsize=fontsize_label)
        plt.show()



"""
ax.errorbar([1,2,3,4],[mean1, mean2, mean3, mean4], yerr=[std1, std2, std3, std4], ls='', capsize=10, fmt='o')
ax.errorbar([1,2,3,4],[1,2,3,4], ls='--')
ax.set_title('Diffusion coefficients',fontsize=18)
ax.set_xlabel('Ground truth', fontsize=18)
ax.set_ylabel('Prediction', fontsize=18)
ax.set_xlim([0.8,4.2])
ax.set_ylim([0.8,4.2])
plt.show()

validation_k_arr = np.squeeze(validation_k.data.numpy(), axis=1)
validation_T_arr = np.squeeze(validation_T.data.numpy(), axis=1)

k_ratios_gt = np.arange(1,4,1)
diff_coef_ratios_gt = np.arange(2,10,2)
np.tile(k_ratios_gt[:, np.newaxis], (1, len(diff_coef_ratios_gt))).shape
k_ratios_gt_tiled = np.tile(k_ratios_gt[:, np.newaxis], (1, len(diff_coef_ratios_gt))).flatten()
diff_coef_ratios_gt_tiled = np.tile(diff_coef_ratios_gt[np.newaxis, :], (len(k_ratios_gt), 1)).flatten()

#diff_coef_ratios_gt_label = np.tile(diff_coef_ratios_gt_tiled.flatten()[:, np.newaxis], (1, n_data_val)).flatten()
#k_ratios_gt_label = np.tile(k_ratios_gt_tiled.flatten()[:, np.newaxis], (1, n_data_val)).flatten()



validation_k_arr_std = validation_k_arr.reshape([-1, n_data_val]).std(axis=1)
validation_k_arr_mean = validation_k_arr.reshape([-1, n_data_val]).mean(axis=1)

validation_T_arr_std = validation_T_arr.reshape([-1, n_data_val]).std(axis=1)
validation_T_arr_mean = validation_T_arr.reshape([-1, n_data_val]).mean(axis=1)
'''
fig, ax = plt.subplots()
for i in range(12):
    ax.errorbar([validation_k_arr_mean[i], k_ratios_gt_tiled[i]], [validation_T_arr_mean[i], diff_coef_ratios_gt_tiled[i]], xerr=[validation_k_arr_std[i], 0], yerr=[validation_T_arr_std[i], 0], fmt='o', ls='-', capsize=10)
    #ax.errorbar([validation_k_arr_mean[i], k_ratios_gt_tiled[i]], [validation_T_arr_mean[i], diff_coef_ratios_gt_tiled[i]], xerr=[validation_k_arr_std[i], 1], yerr=[validation_T_arr_std[i], 1])


plt.show()
'''

from matplotlib.pyplot import cm
color=cm.rainbow(np.linspace(0,1,12))

fig, ax = plt.subplots()
for i in range(12):
    ax.errorbar(validation_k_arr_mean[i], validation_T_arr_mean[i], xerr=validation_k_arr_std[i], yerr=validation_T_arr_std[i], fmt='o', capsize=10, c=color[i])
    ax.plot([validation_k_arr_mean[i], k_ratios_gt_tiled[i]], [validation_T_arr_mean[i], diff_coef_ratios_gt_tiled[i]], marker='o', color=color[i])
    #ax.errorbar([validation_k_arr_mean[i], k_ratios_gt_tiled[i]], [validation_T_arr_mean[i], diff_coef_ratios_gt_tiled[i]], xerr=[validation_k_arr_std[i], 1], yerr=[validation_T_arr_std[i], 1])

ax.set_ylabel('Temperature ratio', fontsize=18)
ax.set_xlabel('Spring constant ratio', fontsize=18)

plt.show()
"""










