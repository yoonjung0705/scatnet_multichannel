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
ROOT_DIR = './data/simulations/'

# common inputs
data_len = 2**8
avg_lens = [2**4, 2**6]
n_filter_octaves = [(1,1)]
dt = 0.001
n_datas = [100, 300]
n_data_test = 300

root_dir = ROOT_DIR
file_names_data = []
file_names_scat = []

k_ratios = [1., 2., 3., 4.]
#k_ratios = [4., 5., 6., 7.]
diff_coef_ratios = [4., 5., 6., 7.]

k_ratios_test_low = 1.3
k_ratios_test_high = 3.7
diff_coef_ratios_test_low = 4.3
diff_coef_ratios_test_high = 6.7

# simulate two beads
for n_data in n_datas:
    print("simulating training data for n_data:{}".format(n_data))
    file_name_data = siu.sim_two_beads(data_len, k_ratios=k_ratios, diff_coef_ratios=diff_coef_ratios, dt=dt, n_data=n_data, n_steps_initial=10000, save_file=True, root_dir=root_dir)
    file_names_data.append(file_name_data)
    for avg_len in avg_lens:
        for n_filter_octave in n_filter_octaves:
            try:
                print("\tscat transforming with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
                file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
                file_names_scat.append(file_name_scat)
            except:
                print("\texception occurred with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))

# simulate data for testing performance
print("simulating test data for n_data_test:{}".format(n_data_test))
k_ratios_test = (k_ratios_test_high - k_ratios_test_low) * np.random.random(n_data_test,) + k_ratios_test_low 
diff_coef_ratios_test = (diff_coef_ratios_test_high - diff_coef_ratios_test_low) * np.random.random(n_data_test,) + diff_coef_ratios_test_low
k_ratios_diff_coef_ratios_test = np.stack([k_ratios_test, diff_coef_ratios_test], axis=1)

data_tests = []
for k_ratio_test, diff_coef_ratio_test in k_ratios_diff_coef_ratios_test:
    data_test = siu.sim_two_beads(data_len, k_ratios=k_ratio_test, diff_coef_ratios=diff_coef_ratio_test, dt=dt, n_data=1, n_steps_initial=10000, save_file=False)
    data_tests.append(data_test)
processes = np.concatenate(data_tests, axis=2) # shaped (1, 1, n_data_test, n_channels, data_len)
samples = {'data':processes, 'labels':k_ratios_diff_coef_ratios_test, 'label_names':'k_ratios_diff_coef_ratios', 'dt':dt, 'n_steps_initial':10000}
nums = cu.match_filename(r'tbd_([0-9]+)_test.pt', root_dir=root_dir)
nums = [int(num) for num in nums]
idx = max(nums) + 1 if nums else 0

file_name_test = 'tbd_{}_test.pt'.format(idx)
file_path_test = os.path.join(root_dir, file_name_test)
torch.save(samples, file_path_test)

# scat transforming test data
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("\tscat transforming with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
            file_name_scat_test = scu.scat_transform(file_name_test, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
        except:
            print("\texception occurred with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
