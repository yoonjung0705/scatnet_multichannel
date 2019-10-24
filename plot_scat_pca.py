'''
visualization scattering transform results using PCA
'''

import numpy as np
import sim_utils as siu
import scat_utils as scu
import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

#plt.style.use('dark_background')

# scatnet parameters
data_len = 2**11
avg_len = 2**8
n_data = 200
dt = 0.001
n_filter_octave = [1,1]
sim_type = 'tbd' # 'brw', 'psn', 'obd', 'tbd'

n_decim = 3 # precision for external parameters
scat = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)

# simulate brownian
if sim_type == 'brw':
    diff_coefs_brw = np.arange(4,8,1)
    traj_brw = siu.sim_brownian(data_len, diff_coefs_brw, dt=dt, n_data=n_data)
    scat_brw = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    traj_brw = traj_brw.reshape(-1, 1, traj_brw.shape[-1])
    S_brw = scat.transform(traj_brw)
    S_brw_log = scu.log_scat(S_brw)
    S_brw_log = scu.stack_scat(S_brw_log)
    S_brw_log_mean = S_brw_log.mean(axis=-1) # average along time axis
    S_brw_log_mean = np.reshape(S_brw_log_mean, (S_brw_log_mean.shape[0], -1))

    diff_coefs_brw_str = np.round(diff_coefs_brw, n_decim).astype(str)
    labels_brw = np.repeat(diff_coefs_brw_str[:, np.newaxis], n_data, axis=-1)
    labels_brw = np.char.add('brw_D', labels_brw)
    labels_brw = labels_brw.flatten()

    X = S_brw_log_mean
    labels = labels_brw

# simulate poisson
elif sim_type == 'psn':
    lams_psn = np.arange(4,8,1)
    traj_psn = siu.sim_poisson(data_len, lams_psn, dt=dt, n_data=n_data)
    scat_psn = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    traj_psn = traj_psn.reshape(-1, 1, traj_psn.shape[-1])
    S_psn = scat.transform(traj_psn)
    S_psn_log = scu.log_scat(S_psn)
    S_psn_log = scu.stack_scat(S_psn_log)
    S_psn_log_mean = S_psn_log.mean(axis=-1)
    S_psn_log_mean = np.reshape(S_psn_log_mean, (S_psn_log_mean.shape[0], -1))

    lams_psn_str = np.round(lams_psn, n_decim).astype(str)
    labels_psn = np.repeat(lams_psn_str[:, np.newaxis], n_data, axis=-1)
    labels_psn = np.char.add('psn_lam', labels_psn)
    labels_psn = labels_psn.flatten()

    X = S_psn_log_mean
    labels = labels_psn

# simulate one bead
elif sim_type == 'obd':
    diff_coefs_obd = np.arange(3,6,1)
    ks_obd = np.arange(2,4,1)
    traj_obd = siu.sim_one_bead(data_len, diff_coefs=diff_coefs_obd, ks=ks_obd, dt=dt, n_data=n_data)
    scat_obd = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    traj_obd = traj_obd.reshape(-1, 1, traj_obd.shape[-1])
    S_obd = scat.transform(traj_obd)
    S_obd_log = scu.log_scat(S_obd)
    S_obd_log = scu.stack_scat(S_obd_log)
    S_obd_log_mean = S_obd_log.mean(axis=-1)
    S_obd_log_mean = np.reshape(S_obd_log_mean, (S_obd_log_mean.shape[0], -1))

    diff_coefs_obd_mesh, ks_obd_mesh = np.meshgrid(diff_coefs_obd, ks_obd, indexing='ij')

    diff_coefs_obd_str = np.round(diff_coefs_obd_mesh, n_decim).astype(str)
    labels_diff_coefs_obd = np.repeat(np.expand_dims(diff_coefs_obd_str, axis=-1), n_data, axis=-1)
    labels_diff_coefs_obd = np.char.add('_D', labels_diff_coefs_obd)

    ks_obd_str = np.round(ks_obd_mesh, n_decim).astype(str)
    labels_ks_obd = np.repeat(np.expand_dims(ks_obd_str, axis=-1), n_data, axis=-1)
    labels_ks_obd = np.char.add('_k', labels_ks_obd)

    labels_obd = np.char.add(labels_diff_coefs_obd, labels_ks_obd)
    labels_obd = np.char.add('obd', labels_obd)
    labels_obd = labels_obd.flatten()

    X = S_obd_log_mean
    labels = labels_obd

# simulation two beads
elif sim_type == 'tbd':
    diff_coef_ratios = np.arange(3,9,2)
    k_ratios = np.arange(2,6,2)
    traj_tbd = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data) 
    traj_tbd = traj_tbd.reshape(-1,2,traj_tbd.shape[-1])
    scat_tbd = scu.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave) 
    S_tbd = scat.transform(traj_tbd)
    S_tbd_log = scu.log_scat(S_tbd)
    S_tbd_log = scu.stack_scat(S_tbd_log)   
    S_tbd_log_mean = S_tbd_log.mean(axis=-1)
    S_tbd_log_mean = np.reshape(S_tbd_log_mean, (S_tbd_log_mean.shape[0], -1))

    diff_coef_ratios_mesh, k_ratios_mesh = np.meshgrid(diff_coef_ratios, k_ratios, indexing='ij')

    diff_coef_ratios_str = np.round(diff_coef_ratios_mesh, n_decim).astype(str)
    labels_diff_coef_ratios = np.repeat(np.expand_dims(diff_coef_ratios_str, axis=-1), n_data, axis=-1)
    labels_diff_coef_ratios = np.char.add('_D', labels_diff_coef_ratios)

    k_ratios_str = np.round(k_ratios_mesh, n_decim).astype(str)
    labels_k_ratios = np.repeat(np.expand_dims(k_ratios_str, axis=-1), n_data, axis=-1)
    labels_k_ratios = np.char.add('_k', labels_k_ratios)

    labels_tbd = np.char.add(labels_diff_coef_ratios, labels_k_ratios)
    labels_tbd = np.char.add('obd', labels_tbd)
    labels_tbd = labels_tbd.flatten()

    X = S_tbd_log_mean
    labels = labels_tbd

else:
    raise("sim_type should be either 'brw', 'psn', 'obd', 'tbd'")

labels_uniq = np.unique(labels)
y = labels

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

fig,ax = plt.subplots()
for label in labels_uniq:
    X_label = X[label==labels, :]
    ax.scatter(X_label[:, 0], X_label[:, 1], 4, label=label)

ax.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.show()

