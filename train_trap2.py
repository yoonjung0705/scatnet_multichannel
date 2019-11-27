'''module that processes the optical trap passive experiment data into the form for scat transform and LSTM training'''
import os
import numpy as np
import pandas as pd
import scat_utils as scu
import sim_utils as siu
import net_utils as nu
import torch
import glob
from itertools import product

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/experiment/position_stiffness'

# common inputs
data_len = 2**9
root_dir = ROOT_DIR
train_test_ratio = [0.6, 0.2]

file_names_data = glob.glob(os.path.join(root_dir, 'L_*.csv'))

file_data_lens = []
# determine number of timepoints per condition in case it differs among files
for file_name_data in file_names_data:
    data = pd.read_csv(file_name_data, header=None)
    file_data_lens.append(len(data))

data_len_total = min(file_data_lens) - 1 # the first element is the stiffness
print("Data length of the files:{}, taking {} timepoints for each file.".format(file_data_lens, data_len_total))
    k = data.loc[0, 0]
    data = data.loc[1:, 0].values
    
# load and split data
for file_name_data in file_names_data:
    data = pd.read_csv(file_name_data, header=None)
    k = data.loc[0, 0]
    data = data.loc[1:data_len_total + 1, 0].values
    data_len_total = data_len_total // data_len * data_len
    data = data[:data_len_total].reshape([-1, 1, data_len]) # shaped [n_data, 1, data_len]
    datas.append(data)
    labels.append(k)

data = np.stack(datas, axis=0) # shaped (n_conditions, n_data, 1, data_len)
samples = {'data':data, 'label_names':['k'], 'labels':labels}
        
torch.save(samples, os.path.join(root_dir, 
        file_name_data = siu.sim_two_beads(data_len, k_ratios, diff_coef_ratios, dt, n_data, n_steps_initial=10000, save_file=True, root_dir=root_dir)
        file_names_data.append(file_name_data)
        for avg_len in avg_lens:
            for n_filter_octave in n_filter_octaves:
                try:
                    print("scat transforming n_data:{} with parameters avg_len:{}, n_filter_octave:{}".format(n_data, avg_len, n_filter_octave))
                    file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
                    file_names_scat.append(file_name_scat)
                except:
                    pass

# simulate data for testing performance
print("simulating data for evaluation")
file_name_test_data_orig = siu.sim_two_beads(data_len, k_ratios_test, diff_coef_ratios_test, dt, n_data_test, n_steps_initial=10000, save_file=True, root_dir=root_dir)
nums = cu.match_filename(r'tbd_([0-9]+)_test.pt', root_dir=root_dir)
nums = [int(num) for num in nums]; idx = max(nums) + 1 if nums else 0
file_name_test_data = 'tbd_{}_test.pt'.format(idx)
os.rename(os.path.join(root_dir, file_name_test_data_orig), os.path.join(root_dir, file_name_test_data))
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("scat transforming n_data:{} with parameters avg_len:{}, n_filter_octave:{} for testing".format(n_data_test, avg_len, n_filter_octave))
            file_name_test_scat_orig = scu.scat_transform(file_name_test_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
        except:
            pass

# train RNNs for scat transformed data
for file_name_scat in file_names_scat:
    meta = torch.load(os.path.join(root_dir, file_name_scat))
    avg_len = meta['avg_len']
    n_filter_octave = meta['n_filter_octave']
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"\
                        .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_scat, [hidden_size, hidden_size], n_layers, bidirectional,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir)
                except:
                    pass


'''
# train RNNs for raw data
for file_name_data in file_names_data:
    for hidden_size in hidden_sizes:
        for n_layers in n_layerss:
            for bidirectional in bidirectionals:
                try:
                    print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
                    nu.train_rnn(file_name_data, [hidden_size, hidden_size], n_layers, bidirectional,
                        n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
                        n_workers=n_workers, root_dir=root_dir)
                except:
                    pass
'''

