'''this is a module that loads the meta data of the neural net (FC and/or RNN) and generates figures that compare/show the performance'''
# FIXME: for now we test prediction results for data simulated with hyperparameters that were used for the training data.
# Later, include data simulated with hyperparameters whose values were not the ones used for the training data

'''standard imports'''
import os
import torch
import matplotlib.pyplot as plt

'''custom libraries'''
import scat_utils as scu
import net_utils as nu

# plt.style.use('dark_background')
file_names = ['tbd_1_scat.pt', 'tbd_1.pt']
file_names_meta = ['tbd_0_scat_meta_rnn_2.pt', 'tbd_0_meta_rnn_1.pt']
root_dir = './data/'
epochs = [50, 40]

'''sanity check'''
# add .pt extension if not provided
file_names = [os.path.splitext(file_name)[0] + '.pt' for file_name in file_names]
file_names_meta = [os.path.splitext(file_name_meta)[0] + '.pt' for file_name_meta in file_names_meta]

# make file_names into list
if isinstance(file_names, str):
    file_names = [file_names]

# check number of files match with number of meta files
assert(len(file_names) == len(file_names_meta))

# check if all meta files are trained on the same dataset
file_names_meta_common = [re.fullmatch(r'([a-z]{3}_[0-9]+)_.*_meta', file_name_meta).group(1) for file_name_meta in file_names_meta]
assert(len(set(file_name_common)) == 1), "The given neural networks are trained on different data sets"

# check if all meta files are testing on the same dataset
file_names_common = [re.fullmatch(r'([a-z]{3}_[0-9]+).*', file_name).group(1) for file_name in file_names]
assert(len(set(file_names_common)) == 1), "The given neural networks are tested on different data sets"

# if the training was done on scat transformed data, it should be tested on scat transformed data, too.
# if the training was done on pure time series data, it should be tested on pure time series data, too.
file_names_meta_is_scat = [re.fullmatch(r'[a-z]{3}_[0-9]+_?(.*)_?meta.*', file_name_meta).group(1) for file_name_meta in file_names_meta]
file_names_is_scat = [re.fullmatch(r'[a-z]{3}_[0-9]+_?(.*)\.pt', file_name_meta).group(1) for file_name_meta in file_names_meta]
assert(file_names_meta_is_scat == file_names_is_scat), "The training and test data are not transformed in the same manner"

file_paths = [os.path.join(root_dir, file_name) for file_name in file_names]
idx_epochs = [idx_epoch // 10 for idx_epoch in idx_epochs]

for file_path in file_paths:
    #fig, ax = plt.subplots()
    meta = torch.load(file_path)
    n_labels = len(meta['label_names'])
    if 'meta_rnn' in file_path:
        for idx_label in range(n_labels):
            net = nu.RNN(input_size=meta['input_size'], hidden_size=meta['hidden_size'], n_layers=meta['n_layers'], bidirectional=meta['bidirectional'])
            net.load_state_dict(meta['weights'][idx_label][idx_epochs[idx_label]], map_location='cpu')










    elif 'meta_nn' in file_path:
    else:
        raise IOError("Invalid file name: should be meta data of the trained network")



    torch.load(


output_arr = output.data.numpy()
target_arr = output.data.numpy()
output_arr = np.squeeze(output_arr, axis=1)
target_arr = np.squeeze(target_arr, axis=1)
output_arr1 = output_arr[:1000]
output_arr2 = output_arr[1000:2000]
output_arr3 = output_arr[2000:3000]
output_arr4 = output_arr[3000:]
std1 = output_arr1.std()
std2 = output_arr2.std()
std3 = output_arr3.std()
std4 = output_arr4.std()
mean1 = output_arr1.mean()
mean2 = output_arr2.mean()
mean3 = output_arr3.mean()
mean4 = output_arr4.mean()
ax.errorbar([1,2,3,4],[mean1, mean2, mean3, mean4], yerr=[std1, std2, std3, std4], ls='', capsize=10, fmt='o')
ax.errorbar([1,2,3,4],[1,2,3,4], ls='--')
ax.set_title('Diffusion coefficients',fontsize=18)
ax.set_xlabel('Ground truth', fontsize=18)
ax.set_ylabel('Prediction', fontsize=18)
ax.set_xlim([0.8,4.2])
ax.set_ylim([0.8,4.2])
plt.show()






fig, ax = plt.subplots()
ax.plot(loss_arr_k, label='Training error')
ax.plot(val_arr_k, label='Test error')
ax.set_title('Inferring spring constant ratio', fontsize=18)
ax.set_ylabel('Mean squared error', fontsize=18)
ax.set_xlabel('Epochs', fontsize=18)
ax.legend()

fig2, ax2 = plt.subplots()
ax2.plot(loss_arr_T, label='Training error')
ax2.plot(val_arr_T, label='Test error')
ax2.set_title('Inferring temperature ratio', fontsize=18)
ax2.set_ylabel('Mean squared error', fontsize=18)
ax2.set_xlabel('Epochs', fontsize=18)
ax2.legend()












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











