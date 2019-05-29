plt.style.use('dark_background')
fig, ax = plt.subplots()
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











