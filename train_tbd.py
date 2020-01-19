'''module that simulates the two beads data and trains the LSTM and scat-transform + LSTM model'''
import os
import numpy as np
import scat_utils as scu
import sim_utils as siu
import net_utils as nu
import torch
from itertools import product

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/simulations/two_beads/'

# common inputs
data_len = 2**8
avg_lens = [2**6]
n_filter_octaves = [(1,1)]
#n_filter_octaves = list(product([1,2,4,8], [1,2,4,8]))
# [(1,1), (1,2), (1,4), (2,1), (2,2), (2,4), (4,1), (4,2), (4,4)]
dt = 0.0001
#n_datas = [50, 100, 200]
n_datas = [100]
n_data_test = 50

root_dir = ROOT_DIR
file_names_data = []
file_names_scat = []

n_epochs_max = 1000
train_ratio = 0.8
batch_size = 20
n_workers = 4

# NN inputs
#n_nodes_hiddens = [] # FIXME: add later

# RNN inputs
hidden_sizes = [20]
n_layerss = [2]
bidirectionals = [True]

k_ratios = [1., 2., 3.]
#k_ratios = np.arange(1,12,1)
diff_coef_ratios = [1., 2., 3., 4.]
#diff_coef_ratios = np.arange(3,8,1)

k_ratios_test_low = 1.5
k_ratios_test_high = 2.5
diff_coef_ratios_test_low = 1.5
diff_coef_ratios_test_high = 3.5

# simulate two beads
print("simulating data for training")
for n_data in n_datas:
    file_name_data = siu.sim_two_beads(data_len, k_ratios=k_ratios, diff_coef_ratios=diff_coef_ratios, dt=dt, n_data=n_data, n_steps_initial=10000, save_file=True, root_dir=root_dir)
    file_names_data.append(file_name_data)
    for avg_len in avg_lens:
        for n_filter_octave in n_filter_octaves:
            try:
                print("scat transforming n_data:{} with parameters avg_len:{}, n_filter_octave:{}".format(n_data, avg_len, n_filter_octave))
                file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
                file_names_scat.append(file_name_scat)
            except:
                print("exception occurred during scat transformation for n_data:{} with parameters avg_len:{}, n_filter_octave:{}".format(n_data, avg_len, n_filter_octave))

# simulate data for testing performance
print("simulating data for evaluation for randomly sampled labels")
k_ratios_test = (k_ratios_test_high - k_ratios_test_low) * np.random.random(n_data_test,) + k_ratios_test_low 
diff_coef_ratios_test = (diff_coef_ratios_test_high - diff_coef_ratios_test_low) * np.random.random(n_data_test,) + diff_coef_ratios_test_low
k_ratios_diff_coef_ratios_test = np.stack([k_ratios_test, diff_coef_ratios_test], axis=1)



#### continue here

file_name_test_data = siu.sim_two_beads(data_len, k_ratios=k_ratios_test, diff_coef_ratios=diff_coef_ratios_test, dt=dt, n_data=n_data_test, n_steps_initial=10000, save_file=True, root_dir=root_dir)
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("scat transforming n_data:{} with parameters avg_len:{}, n_filter_octave:{} for testing".format(n_data_test, avg_len, n_filter_octave))
            file_name_test_scat = scu.scat_transform(file_name_test_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
        except:
            print("exception occurred during scat transformation for parameters avg_len:{}, n_filter_octave:{}".format(n_data, avg_len, n_filter_octave))

# train RNNs for scat transformed data
for file_name_scat in file_names_scat:
    meta = torch.load(os.path.join(root_dir, file_name_scat))
    avg_len = meta['avg_len']
    n_filter_octave = meta['n_filter_octave']
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_scat, [hidden_size, hidden_size], n_layers, bidirectional, classifier=False,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir)
                except:
                    print("exception occurred during training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))


'''
# train RNNs for raw data
for file_name_data in file_names_data:
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_data, [hidden_size, hidden_size], n_layers, bidirectional, classifier=False,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir)
                except:
                    print("exception occurred during training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}"\
                        .format(file_name_data, hidden_size, n_layers, bidirectional))
'''
