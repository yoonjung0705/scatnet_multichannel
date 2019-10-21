import numpy as np
import scat_utils as scu
import sim_utils as siu
import net_utils as nu

ROOT_DIR = './data/'

# common inputs
data_len = 2**11
avg_len = 2**8
n_filter_octave = [1, 1]
dt = 0.01 # REVIEW
n_datas = [20, 100, 500]

root_dir = ROOT_DIR
file_names_data = []
file_names_scat = []

# NN inputs
#n_nodes_hiddens = [] # FIXME: add later
n_epochs_max = 1000
train_ratio = 0.8
batch_size = 100
n_workers = 4

# RNN inputs
hidden_sizes = [5, 20, 50, 100, 200]
n_layerss = [1, 2]
bidirectionals = [True, False]

"""
# simulate brownian
diff_coefs = np.arange(4,8, 0.5)
siu.sim_brownian(data_len, diff_coefs, dt, n_data, save_file=True, root_dir=ROOT_DIR)

# simulate one bead
diff_coefs = np.arange(4,8, 0.5)
ks = np.arange(1,3,0.5)
siu.sim_one_bead(data_len, diff_coefs, ks, dt, n_data, n_steps_initial=10000, save_file=True, root_dir=ROOT_DIR)

# simulate poisson
lams = np.arange(4,8, 0.5)
siu.sim_poisson(data_len, lams, dt, n_data, save_file=True, root_dir=ROOT_DIR)
"""

# simulate two beads
k_ratios = [1, 2, 4, 8, 16]
diff_coef_ratios = [1, 2, 4, 8, 16]
for n_data in n_datas:
    print("simulating data for n_data:{}".format(n_data))
    file_name_data = siu.sim_two_beads(data_len, k_ratios, diff_coef_ratios, dt, n_data, n_steps_initial=10000, save_file=True, root_dir=root_dir)
    file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
    file_names_data.append(file_name_data)
    file_names_scat.append(file_name_scat)

# train RNNs for scat transformed data
for file_name_scat in file_names_scat:
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_scat, hidden_size, n_layers, bidirectional))
                nu.train_rnn(file_name_scat, [hidden_size, hidden_size], n_layers, bidirectional,
                    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                    n_workers=n_workers, root_dir=root_dir)


