'''
visualization after scattering transform of experimental data using PCA
experimental data being either 
- beads with optical trap (one bead)
- beads without optical trap (multiple beads)
- irfp data: in this case, we pick a subset of trajectories that are long enough for visualization since the trajectories have different length

NOTES: compared to the plot_scat_pca_sim.py, the difference is that there's no simulation done here.
We'll be using existing data, so we need to provide the file path

When visualizing the scat transformed data, need to check if it's log transformed. If not, need to take the log

TODO: only did for beads. merge one bead and multiple beads case, only separate out the irfp case
I need to match the color between labels of scat and raw. Maybe I need to generate scat plot first, then get the colors for each label, then use that for the raw. But if the plot comes out as the same color paired, then it should be fine.
I realized that most plots are missing (empty plots). figure out why this is happening

NOTE: for marker sizes, we set it to be the same for all dots. This is because 
'''

import numpy as np
import sim_utils as siu
import scat_utils as scu
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
#from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
#plt.style.use('dark_background')
fontsize_title = 18
fontsize_label = 14
fontsize_legend = 14
exp_type = 'obd' # 'obd', 'mbd', 'irfp'. experiment type
marker_size_mean = 13

n_decim = 3 # precision for external parameters

# one bead (with optical trap)
if exp_type == 'obd':
    #root_dir = '/home/yoonjung/repos/scatnet_multichannel/data/scat_experimental_data_for_visualization/beads_with_optical_trap_2020_0305'
    root_dir = '/home/yoonjung/repos/scatnet_multichannel/data/scat_experimental_data_for_visualization/beads_without_optical_trap_2020_0319'
    file_name_raw = 'data.pt'
    file_name_scat = 'data_scat_0.pt'
    file_path_raw = os.path.join(root_dir, file_name_raw)
    file_path_scat = os.path.join(root_dir, file_name_scat)

    data_raw = torch.load(file_path_raw)
    data_scat = torch.load(file_path_scat)

    # check if labels are identical and all trajectories are in corresponding order
    assert(data_raw['labels'] == data_scat['labels']), "Labels between the raw data and the scat transformed data are not identical"

    # log transform the scat transformed data's absolute values if it's not log transformed yet
    if not data_scat['log_transform']:
        # in this case, we will NOT use the data_scat's data directly
        # this is because data_scat has already gone through stack_scat which is irreversible
        # (lose a bit of meta data)
        # so let's fetch the data_scat's scat transform hyperparameters, use that for doing scat transform from scratch, then visualize it
        avg_len = data_scat['avg_len']
        n_filter_octave = data_scat['n_filter_octave']
        data_len = data_raw['data'].shape[-1]
        #samples = siu.sim_brownian(data_len, diff_coefs_brw, dt=dt, n_data=n_data)
        #traj_brw = samples['data']
        scat = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
        S = scat.transform(data_raw['data'])
        S_log = scu.log_scat(S)
        S_log = scu.stack_scat(S_log)
    else:
        S_log = data_scat['data']

    S_log_mean = S_log.mean(axis=-1) # average along time axis
    S_log_mean = np.reshape(S_log_mean, (S_log_mean.shape[0], -1))

    # create labels as ndarray shaped (n_trajectories,)
    # TODO: continue here
    labels_raw_idx = np.array(data_raw['labels_lut'])[data_raw['labels']]
    labels_scat_idx = np.array(data_scat['labels_lut'])[data_scat['labels']]
    n_data = len(labels_raw_idx)
    labels_raw = np.array([''] * n_data)
    labels_scat = np.array([''] * n_data)
    for label_idx, label_name in enumerate(data_raw['label_names']):
        if label_idx > 0:
            labels_raw = np.char.add(labels_raw, ', ')
            labels_scat = np.char.add(labels_scat, ', ')
        labels_raw = np.char.add(labels_raw, np.char.add(label_name + '=', labels_raw_idx[:, label_idx].astype(str)))
        labels_scat = np.char.add(labels_scat, np.char.add(label_name + '=', labels_scat_idx[:, label_idx].astype(str)))

    
    X_scat = S_log_mean
    X_raw = np.reshape(data_raw['data'], [n_data, -1])
    
# simulate poisson
elif sim_type == 'psn':
    lams_psn = np.arange(4,8,1)
    samples = siu.sim_poisson(data_len, lams_psn, dt=dt, n_data=n_data)
    traj_psn = samples['data']
    scat_psn = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    S_psn = scat.transform(traj_psn)
    S_psn_log = scu.log_scat(S_psn)
    S_psn_log = scu.stack_scat(S_psn_log)
    S_psn_log_mean = S_psn_log.mean(axis=-1)
    S_psn_log_mean = np.reshape(S_psn_log_mean, (S_psn_log_mean.shape[0], -1))

    lams_psn_str = np.round(lams_psn, n_decim).astype(str)
    labels_psn = np.repeat(lams_psn_str[:, np.newaxis], n_data, axis=-1)
    labels_psn = np.char.add('lambda=', labels_psn)
    labels_psn = labels_psn.flatten()

    X = S_psn_log_mean
    labels = labels_psn

# simulate one bead
elif sim_type == 'obd':
    diff_coefs_obd = np.arange(3,6,1)
    ks_obd = np.arange(2,4,1)
    samples = siu.sim_one_bead(data_len, ks=ks_obd, diff_coefs=diff_coefs_obd, dt=dt, n_data=n_data)
    traj_obd = samples['data']
    scat_obd = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    S_obd = scat.transform(traj_obd)
    S_obd_log = scu.log_scat(S_obd)
    S_obd_log = scu.stack_scat(S_obd_log)
    S_obd_log_mean = S_obd_log.mean(axis=-1)
    S_obd_log_mean = np.reshape(S_obd_log_mean, (S_obd_log_mean.shape[0], -1))

    diff_coefs_obd_mesh, ks_obd_mesh = np.meshgrid(diff_coefs_obd, ks_obd, indexing='ij')

    diff_coefs_obd_str = np.round(diff_coefs_obd_mesh, n_decim).astype(str)
    labels_diff_coefs_obd = np.repeat(np.expand_dims(diff_coefs_obd_str, axis=-1), n_data, axis=-1)
    labels_diff_coefs_obd = np.char.add('D=', labels_diff_coefs_obd)

    ks_obd_str = np.round(ks_obd_mesh, n_decim).astype(str)
    labels_ks_obd = np.repeat(np.expand_dims(ks_obd_str, axis=-1), n_data, axis=-1)
    labels_ks_obd = np.char.add(', k=', labels_ks_obd)

    labels_obd = np.char.add(labels_diff_coefs_obd, labels_ks_obd)
    #labels_obd = np.char.add('obd', labels_obd)
    labels_obd = labels_obd.flatten()

    X = S_obd_log_mean
    labels = labels_obd

# simulation two beads
elif sim_type == 'tbd':
    gammas = np.arange(1,1.4,0.05) # in paper it's [1, 3]
    #k_ratios = np.arange(2,9,2)
    #k_ratios = [1,3,5,7]
    k_ratios = [4] # in paper it's [1,7]
    #diff_coef_ratios = np.arange(4,11) # in paper it's [4,10]
    diff_coef_ratios = [4,6,8,10] 
    #diff_coef_ratios = np.arange(3,4)
    samples = siu.sim_two_beads(data_len, gammas=gammas, k_ratios=k_ratios, diff_coef_ratios=diff_coef_ratios, dt=dt, n_data=n_data) 
    traj_tbd = samples['data']
    scat_tbd = scu.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave) 
    S_tbd = scat.transform(traj_tbd)
    S_tbd_log = scu.log_scat(S_tbd)
    S_tbd_log = scu.stack_scat(S_tbd_log)   
    S_tbd_log_mean = S_tbd_log.mean(axis=-1)
    S_tbd_log_mean = np.reshape(S_tbd_log_mean, (S_tbd_log_mean.shape[0], -1))
    X_raw = traj_tbd.reshape([traj_tbd.shape[0], -1])

    gammas_mesh, diff_coef_ratios_mesh, k_ratios_mesh = np.meshgrid(gammas, diff_coef_ratios, k_ratios, indexing='ij')

    #gammas_str = np.round(gammas_mesh, n_decim).astype(str)
    labels_gammas = np.repeat(np.expand_dims(gammas_mesh, axis=-1), n_data, axis=-1).flatten()
    labels_diff_coef_ratios = np.repeat(np.expand_dims(diff_coef_ratios_mesh, axis=-1), n_data, axis=-1).flatten()
    labels_k_ratios = np.repeat(np.expand_dims(k_ratios_mesh, axis=-1), n_data, axis=-1).flatten()
    #labels_gammas = np.char.add('\gamma=', labels_gammas)
    marker_sizes = labels_gammas**4
    marker_sizes = marker_sizes / np.mean(marker_sizes) * marker_size_mean

    #diff_coef_ratios_str = np.round(diff_coef_ratios_mesh, n_decim).astype(str)
    #labels_diff_coef_ratios = np.repeat(np.expand_dims(diff_coef_ratios_str, axis=-1), n_data, axis=-1)
    #labels_diff_coef_ratios = np.char.add(r'$T_h/T_c=$', labels_diff_coef_ratios)

    #k_ratios_str = np.round(k_ratios_mesh, n_decim).astype(str)
    #labels_k_ratios = np.repeat(np.expand_dims(k_ratios_str, axis=-1), n_data, axis=-1)
    #labels_k_ratios = np.char.add(r'$, k=$', labels_k_ratios)

    #labels_tbd = np.char.add(labels_diff_coef_ratios, labels_k_ratios)
    #labels_tbd = np.char.add('obd', labels_tbd)
    #labels_tbd = labels_tbd.flatten()

    X = S_tbd_log_mean
    #labels = labels_tbd

    #labels_uniq = np.unique(labels)

    # standardize (for each variable) before pca for both X and X_raw
    scaler = StandardScaler()
    X_scat = scaler.fit_transform(X_scat)
    X_raw = scaler.fit_transform(X_raw)

    pca = decomposition.PCA(n_components=2)
    pca.fit(X_scat)
    X_scat = pca.transform(X_scat)

    pca = decomposition.PCA(n_components=2)
    pca.fit(X_raw)
    X_raw = pca.transform(X_raw)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    # let's get the unique values of the labels
    # since we asserted that the scat and the raw labels are identical, just use the raw labels
    labels_uniqs = np.unique(labels_raw)

    for labels_uniq in labels_uniqs:
        #label = r'$T_h/T_c={}, k_2/k_1={}$'.format(diff_coef_ratio, k_ratio)
        filt = labels_raw == labels_uniq

        ax.scatter(X_scat[filt, 0], X_scat[filt, 1], label=labels_uniq, s=marker_sizes[filt])
        ax2.scatter(X_raw[filt, 0], X_raw[filt, 1], label=labels_uniq, s=marker_sizes[filt])
"""
    for diff_coef_ratio in diff_coef_ratios:
        for k_ratio in k_ratios:
            #label = r'$T_h/T_c={}, k_2/k_1={}$'.format(diff_coef_ratio, k_ratio)
            label = r'$T_h/T_c={}$'.format(diff_coef_ratio)
            filt = labels_diff_coef_ratios == diff_coef_ratio
            filt &= labels_k_ratios == k_ratio
            ax.scatter(X[filt, 0], X[filt, 1], label=label, s=marker_sizes[filt])
            ax2.scatter(X_raw[filt, 0], X_raw[filt, 1], label=label, s=marker_sizes[filt])
"""
    ax.legend(fontsize=fontsize_legend, loc='upper right')
    ax.set_title('PCA after scattering transform', fontsize=fontsize_title)
    ax.set_xlabel('PC1', fontsize=fontsize_label)
    ax.set_ylabel('PC2', fontsize=fontsize_label)

    ax2.legend(fontsize=fontsize_legend, loc='upper right')
    ax2.set_title('PCA after scattering transform', fontsize=fontsize_title)
    ax2.set_xlabel('PC1', fontsize=fontsize_label)
    ax2.set_ylabel('PC2', fontsize=fontsize_label)

    plt.show()

else:
    raise("sim_type should be either 'brw', 'psn', 'obd', 'tbd'")
"""
labels_uniq = np.unique(labels)

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

fig,ax = plt.subplots()
for label in labels_uniq:
    X_label = X[label==labels, :]
    ax.scatter(X_label[:, 0], X_label[:, 1], label=label, s=marker_sizes[label==labels])

ax.legend()
ax.set_title('PCA after scattering transform', fontsize=fontsize_title)
ax.set_xlabel('PC1', fontsize=fontsize_label)
ax.set_ylabel('PC2', fontsize=fontsize_label)

plt.show()
"""
