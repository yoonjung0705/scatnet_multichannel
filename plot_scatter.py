'''
scatter plot of either raw and scattering transformed time series using PCA, tsne, etc
NOTE: this module is only for data with labels sampled from a finite discrete set of values
FIXME: merge with plot_scat_pca.py
FIXME: for tsne, considering allowing PCA before doing tsne
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import decomposition, manifold

# visualization settings
plt.style.use('dark_background')
plt.close('all')
fontsize_title = 18
fontsize_label = 14
legend_location = 'upper left'
marker_size = None # dot size in scatter plot
method = 'pca' # 'pca' or 'tsne'
perplexity = 5 # required if method is 'tsne'
# load data
root_dir = './data/experiments/bead/2020_0305'
save_dir = root_dir + '/figs'
try:
    os.mkdir(save_dir)
except OSError:
    pass
file_name_1 = 'data.pt'
file_name_2 = 'data_scat_0.pt'
file_name_no_ext_1, _ = os.path.splitext(file_name_1)
file_name_no_ext_2, _ = os.path.splitext(file_name_2)
file_path_1 = os.path.join(root_dir, file_name_1)
file_path_2 = os.path.join(root_dir, file_name_2)
samples_1 = torch.load(file_path_1)
samples_2 = torch.load(file_path_2)

X_1 = samples_1['data']
X_2 = samples_2['data']
X_1 = np.reshape(X_1, [X_1.shape[0], -1])
X_2 = np.reshape(X_2, [X_2.shape[0], -1])

y_1 = np.array(samples_1['labels'])
y_2 = np.array(samples_2['labels'])

assert(len(X_1) == len(X_2)), "Invalid shape given: number of trajectories differ for raw and scat"
assert(np.isclose(y_1, y_2).all()), "Invalid labels given: different labels for raw and scat data"
y = y_1
labels_uniq = np.unique(y)
fig, axs = plt.subplots(1,2, figsize=(18,6))
#fig, subplots = plt.subplots(3, 5, figsize=(15, 8))
if method == 'pca':
    pca = decomposition.PCA(n_components=2)
    X_1 = pca.fit_transform(X_1)
    X_2 = pca.fit_transform(X_2) # TODO: check if doing this without redefining pca is ok
else:
    # TODO: check if setting random_state gives consistent results
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity) 
    X_1 = tsne.fit_transform(X_1)
    X_2 = tsne.fit_transform(X_2)

for label in labels_uniq:
    X_label_1 = X_1[label==y, :]
    X_label_2 = X_2[label==y, :]
    axs[0].scatter(X_label_1[:, 0], X_label_1[:, 1], s=marker_size, label=label)
    axs[1].scatter(X_label_2[:, 0], X_label_2[:, 1], s=marker_size, label=label)

for i in range(2):
    axs[i].legend(loc=legend_location)
    title = '{}'.format(method.upper())
    if i == 0: title += ' of raw data'
    if i == 1: title += ' of scat transformed data'
    if method == 'tsne': title += ' using perplexity {}'.format(perplexity)
    axs[i].set_title(title, fontsize=fontsize_title)
    axs[i].set_xlabel('Dim 1', fontsize=fontsize_label)
    axs[i].set_ylabel('Dim 2', fontsize=fontsize_label)

file_name_save = method
if method == 'tsne': file_name_save += '_{}'.format(perplexity)
file_name_save += '_{}_{}.png'.format(file_name_no_ext_1, file_name_no_ext_2)
file_path_save = os.path.join(save_dir, file_name_save)
fig.savefig(file_path_save)
'''
(fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
perplexities = [5, 30, 50, 100]
for perplexity in perplexities:
    tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
    X = tsne.fit_transform(X)
'''



plt.show()

