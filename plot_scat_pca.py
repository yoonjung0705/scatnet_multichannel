'''
demonstration of scattering transform-based classification
trajectories are scatter-plotted based on the first three principal component axes
'''

import numpy as np
import scatnet as scn
import sim_utils as siu
import scatnet_utils as scu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition

plt.style.use('dark_background')
# scatnet parameters
data_len = 2**11
avg_len = 2**8
n_data = 200
n_filter_octave = [1,1]

scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
# precision for external parameters
n_decim = 3

# simulate brownian
diff_coefs_brw = np.arange(4,8,1)
dt = 0.01
traj_brw = siu.sim_brownian(data_len, diff_coefs_brw, dt=dt, n_data=n_data)
scat_brw = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
traj_brw = traj_brw.reshape(-1, 1, traj_brw.shape[-1])
S_brw = scat.transform(traj_brw)
S_brw_log = scu.log_scat(S_brw)
S_brw_log = scu.stack_scat(S_brw_log)
S_brw_log_mean = S_brw_log.mean(axis=-1) # average along time axis
S_brw_log_mean = np.reshape(S_brw_log_mean, (-1, S_brw_log_mean.shape[-1]))

diff_coefs_brw_str = np.round(diff_coefs_brw, n_decim).astype(str)
labels_brw = np.repeat(diff_coefs_brw_str[:, np.newaxis], n_data, axis=-1)
labels_brw = np.char.add('brw_D', labels_brw)
labels_brw = labels_brw.flatten()

# simulate poisson
lams_psn = np.arange(4,8,1)
dt = 0.01
traj_psn = siu.sim_poisson(data_len, lams_psn, dt=dt, n_data=n_data)
scat_psn = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
traj_psn = traj_psn.reshape(-1, 1, traj_psn.shape[-1])
S_psn = scat.transform(traj_psn)
S_psn_log = scu.log_scat(S_psn)
S_psn_log = scu.stack_scat(S_psn_log)
S_psn_log_mean = S_psn_log.mean(axis=-1)
S_psn_log_mean = np.reshape(S_psn_log_mean, (-1, S_psn_log_mean.shape[-1]))

lams_psn_str = np.round(lams_psn, n_decim).astype(str)
labels_psn = np.repeat(lams_psn_str[:, np.newaxis], n_data, axis=-1)
labels_psn = np.char.add('psn_lam', labels_psn)
labels_psn = labels_psn.flatten()

# simulate one bead
diff_coefs_obd = np.arange(4,8,0.5)
ks_obd = np.arange(1,3,0.5)
dt = 0.01
traj_obd = siu.sim_one_bead(data_len, diff_coefs=diff_coefs_obd, ks=ks_obd, dt=dt, n_data=n_data)
scat_obd = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
traj_obd = traj_obd.reshape(-1, 1, traj_obd.shape[-1])
S_obd = scat.transform(traj_obd)
S_obd_log = scu.log_scat(S_obd)
S_obd_log = scu.stack_scat(S_obd_log)
S_obd_log_mean = S_obd_log.mean(axis=-1)
S_obd_log_mean = np.reshape(S_obd_log_mean, (-1, S_obd_log_mean.shape[-1]))

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

labels = np.concatenate([labels_brw, labels_psn, labels_obd], axis=0)
S_log_mean = np.concatenate([S_brw_log_mean, S_psn_log_mean, S_obd_log_mean], axis=0)

# simulation two beads
diff_coef_ratios = np.arange(2,10,2)
k_ratios = [1]
traj_tbd = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data) 
traj_tbd = traj_tbd.reshape(-1,2,traj_tbd.shape[-1])
scat_tbd = scn.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave) 
S_tbd = scat.transform(traj_tbd)
S_tbd_merge = scu.merge_channels(S_tbd)
S_tbd_merge_log = scu.log_scat(S_tbd_merge)
S_tbd_merge_log_stack = scu.stack_scat(S_tbd_merge_log)   
S_tbd_merge_log_stack_mean = S_tbd_merge_log_stack.mean(axis=-1)

# def plot_scat(X, y, n_components=2):
# X = S_log_mean
X = S_brw_log_mean
labels = labels_brw
# labels_uniq = np.unique(labels)
labels_uniq = np.unique(labels)
y = labels

# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

fig,ax = plt.subplots()
for label in labels_uniq:
    X_label = X[label==labels, :]
    ax.scatter(X_label[:, 0], X_label[:, 1], 4, label=label)

ax.legend()
ax.set_xticks([])
ax.set_yticks([])


#for name, label in [(str(diff_coefs[0]), diff_coefs[0]), (str(diff_coefs[1]), diff_coefs[1]), (str(diff_coefs[2]), diff_coefs[2])]:
#    ax.text3D(X[y == label, 0].mean(), X[y == label, 1].mean() + 1.5, X[y == label, 2].mean(), name, horizontalalignment='center', bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results

#pdb.set_trace()
#y = np.choose(y, [1, 2, 0]).astype(np.float)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], label=y)

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

plt.show()






# plot_scat(X=S_log_mean, y=labels)


    #return trajs_diff_coefs



#scat.transform(
#scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
