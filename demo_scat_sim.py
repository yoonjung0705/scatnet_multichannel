import numpy as np
import scatnet as scn
import sim_utils as siu
import scatnet_utils as scu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
import pdb
# scatnet parameters
#data_len = 2**10
#avg_len = 2**8
#n_filter_octave = [1, 1]

# additional simulation parameters
#diff_coefs = [1, 10, 100]
#dt = 0.01
##k = 1
##lamb = 5
#n_sims = 100

## generate trajectories
##siu.onebead(data_len=data_len, diff_coef=diff_coef, k=k, dt=dt)
##siu.poisson(data_len=data_len, lamb=lamb, dt=dt)

def sim_brownian_diff_coefs(data_len=2**11, avg_len=2**9, diff_coefs=[9, 10, 11], dt=0.01, n_sims=1000):
    diff_coefs_labels_orig = list(range(len(diff_coefs)))
    trajs_diff_coefs = []
    for diff_coef in diff_coefs:
        concat_list = []
        for _ in range(n_sims):
            traj = siu.brownian(data_len=data_len, diff_coef=diff_coef, dt=dt)
            concat_list.append(traj)
            trajs = np.stack(concat_list, axis=0)
        trajs_diff_coefs.append(trajs)

    trajs_diff_coefs = np.stack(trajs_diff_coefs, axis=0)
    
    trajs2 = np.reshape(trajs_diff_coefs, [len(diff_coefs) * n_sims, data_len])
    scat = scn.ScatNet(data_len=data_len, avg_len=avg_len, n_filter_octave=[8,1])
    S = scat.transform(trajs2)
    S_stack = scu.stack_scat(S)
    #print((S_stack > 0).sum())
    #print(S_stack.shape)
    S_stack = np.log2(np.abs(S_stack) + 1.e-6)

    concat_list_diff_coef = []
    for diff_coef_label in diff_coefs_labels_orig:                           
        concat_list_diff_coef.append(np.full((n_sims,), fill_value=diff_coef_label))
    diff_coefs_labels = np.concatenate(concat_list_diff_coef, axis=0)

    #X = np.copy(trajs2)
    X = np.copy(S_stack)
    y = np.copy(diff_coefs_labels)
    
    print(X.shape)
    print(y.shape)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    #for name, label in [(str(diff_coefs[0]), diff_coefs[0]), (str(diff_coefs[1]), diff_coefs[1]), (str(diff_coefs[2]), diff_coefs[2])]:
    #    ax.text3D(X[y == label, 0].mean(), X[y == label, 1].mean() + 1.5, X[y == label, 2].mean(), name, horizontalalignment='center', bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results

    #pdb.set_trace()
    #y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,edgecolor='k')

    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])

    plt.show()








    #return trajs_diff_coefs



#scat.transform(
#scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
