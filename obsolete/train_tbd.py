'''module that trains the LSTM and scat-transform + LSTM model given the simulated files'''
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

<<<<<<< HEAD
# common inputs
data_len = 2**11
avg_lens = [2**4, 2**6, 2**8]
n_filter_octaves = [(1,1)]
#n_filter_octaves = list(product([1,2,4,8], [1,2,4,8]))
# [(1,1), (1,2), (1,4), (2,1), (2,2), (2,4), (4,1), (4,2), (4,4)]
dt = 0.001
#n_datas = [50, 100, 200]
n_datas = [100]
n_data_test = 300

=======
'''common inputs'''
>>>>>>> 18a8402a9b721b66ed28bcee5beabe6256a96076
root_dir = ROOT_DIR
file_names_data = ['tbd_0.pt', 'tbd_1.pt']
#file_names_scat = []

n_epochs_max = 2000
train_ratio = 0.8
<<<<<<< HEAD
batch_size = 128
n_workers = 4

# RNN inputs
hidden_sizes = [20, 50, 100, 200]
n_layerss = [2]
bidirectionals = [True]

k_ratios = [1., 2., 3., 4.]
diff_coef_ratios = [4., 5., 6., 7.]

k_ratios_test_low = 1.2
k_ratios_test_high = 3.8
diff_coef_ratios_test_low = 4.2
diff_coef_ratios_test_high = 6.8

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

data_tests = []
for k_ratio_test, diff_coef_ratio_test in k_ratios_diff_coef_ratios_test:
    data_test = siu.sim_two_beads(data_len, k_ratios=k_ratio_test, diff_coef_ratios=diff_coef_ratio_test, dt=dt, n_data=1, n_steps_initial=10000, save_file=False)
    data_tests.append(data_test)
processes = np.concatenate(data_tests, axis=2) # shaped (1, 1, n_data_test, n_channels, data_len)
samples = {'data':processes, 'labels':k_ratios_diff_coef_ratios_test, 'label_names':'k_ratios_diff_coef_ratios', 'dt':dt, 'n_steps_initial':10000}
nums = cu.match_filename(r'tbd_([0-9]+).pt', root_dir=root_dir)
nums = [int(num) for num in nums]
idx = max(nums) + 1 if nums else 0

file_name_test = 'tbd_{}.pt'.format(idx)
file_path_test = os.path.join(root_dir, file_name_test)
torch.save(samples, file_path_test)

# scat transforming test data
file_names_scat_test = []
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("scat transforming n_data_test:{} with parameters avg_len:{}, n_filter_octave:{}".format(n_data_test, avg_len, n_filter_octave))
            file_name_scat_test = scu.scat_transform(file_name_test, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
            file_names_scat_test.append(file_name_scat_test)
        except:
            print("exception occurred during scat transformation for n_data_test:{} with parameters avg_len:{}, n_filter_octave:{}".format(n_data_test, avg_len, n_filter_octave))

=======
batch_size = 64
n_workers = 4

# RNN inputs
hidden_sizes = [20, 50, 100]
n_layerss = [2, 3]
bidirectionals = [True]

'''
>>>>>>> 18a8402a9b721b66ed28bcee5beabe6256a96076
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
                    print("exception occurred for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))

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
                    print("exception occurred for {}, hidden_size:{}, n_layers:{}, bidirectional:{}"\
                        .format(file_name_data, hidden_size, n_layers, bidirectional))
<<<<<<< HEAD
=======

>>>>>>> 18a8402a9b721b66ed28bcee5beabe6256a96076
