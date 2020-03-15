'''module that generates a .pt file whose data is the displacement'''
import os
import numpy as np
import torch
from copy import deepcopy
# FIXME: add part that performs scat transform after generating data
# FIXME: check the results of the generated displacement file

#ROOT_DIR = './data/simulations/trial_1'
ROOT_DIR = './data/experiments/irfp'
#ROOT_DIR = './data/experiments/bead/2020_0228'
#ROOT_DIR = './data/experiments/bead/2020_0305'

#file_names = ['tbd_0.pt', 'tbd_1.pt', 'tbd_2.pt', 'tbd_3.pt', 'tbd_4.pt', 'tbd_0_test.pt']
file_names = ['data.pt', 'data_test.pt']

# common inputs
root_dir = ROOT_DIR

for file_name in file_names:
    file_path = os.path.join(root_dir, file_name)
    samples = torch.load(file_path)
    samples_out = deepcopy(samples)
    del samples_out['data']
    
    if isinstance(samples['data'], np.ndarray):
        assert(len(samples['data'].shape) == 3),\
            "Invalid data shape given. If type is ndarray, should be rank 3"
        samples_out['data'] = np.diff(samples['data'], n=1, axis=-1)
    elif isinstance(samples['data'], list):
        assert(len(samples['data'][0].shape) == 2),\
            "Invalid data shape given. If type is list, elements should be rank 2 ndarrays"
        data = []
        for track in samples['data']:
            data.append(np.diff(track, n=1, axis=-1))
        samples_out['data'] = data
    else:
        raise ValueError("Invalid data given. Type should be either ndarray or list")
    file_name_no_ext, _ = os.path.splitext(file_name)
    file_name_out = file_name_no_ext + '_disp.pt'
    file_path_out = os.path.join(root_dir, file_name_out)
    torch.save(samples_out, file_path_out)