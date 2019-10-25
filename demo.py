import os
import numpy as np
import scat_utils as scu
import sim_utils as siu
import net_utils as nu

ROOT_DIR = './data/'

# common inputs
data_len = 2**11
avg_len = 2**8
n_filter_octave = [1, 1]
dt = 0.0001
n_datas = [20, 50]
#n_datas = [20, 50, 80, 110]
n_data_test = 300

root_dir = ROOT_DIR
file_names_data = []
file_names_scat = []

n_epochs_max = 1000
train_ratio = 0.8
batch_size = 100
n_workers = 4

# NN inputs
#n_nodes_hiddens = [] # FIXME: add later

# RNN inputs
#hidden_sizes = [5, 20, 50, 100, 200]
hidden_sizes = [50]
n_layerss = [1]
#n_layerss = [1]
#bidirectionals = [True, False]
bidirectionals = [True]

k_ratios = [.2, .4, .7, 1., 2., 4., 6., 8., 10., 12., 14., 16., 18.]
diff_coef_ratios = [3.]

k_ratios_test = [.25, .5, 1., 2., 4., 8., 16.]
diff_coef_ratios_test = [3.]

# simulate two beads
for n_data in n_datas:
    print("simulating data for n_data:{}".format(n_data))
    file_name_data = siu.sim_two_beads(data_len, k_ratios, diff_coef_ratios, dt, n_data, n_steps_initial=10000, save_file=True, root_dir=root_dir)
    file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
    file_names_data.append(file_name_data)
    file_names_scat.append(file_name_scat)

# simulate data for testing performance
print("simulating data for evaluation")
file_name_test_data = siu.sim_two_beads(data_len, k_ratios_test, diff_coef_ratios_test, dt, n_data_test, n_steps_initial=10000, save_file=True, root_dir=root_dir)
file_name_test_scat = scu.scat_transform(file_name_test_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
os.rename(os.path.join(root_dir, file_name_test_data), os.path.join(root_dir, os.path.splitext(file_name_test_data)[0] + '_test.pt'))
os.rename(os.path.join(root_dir, file_name_test_scat), os.path.join(root_dir, os.path.splitext(file_name_test_scat)[0] + '_test.pt'))

# train RNNs for scat transformed data
for file_name_data, file_name_scat in zip(file_names_data, file_names_scat):
#for file_name_data in file_names_data:
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_scat, hidden_size, n_layers, bidirectional))
                nu.train_rnn(file_name_scat, [hidden_size, hidden_size], n_layers, bidirectional,
                    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                    n_workers=n_workers, root_dir=root_dir)

                print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
                nu.train_rnn(file_name_data, [hidden_size, hidden_size], n_layers, bidirectional,
                    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                    n_workers=n_workers, root_dir=root_dir)

