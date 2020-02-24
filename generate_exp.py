'''module that processes the experimental data to later allow training LSTM and scat-transform + LSTM models'''
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
avg_lens = [2**4, 2**6]
n_filter_octaves = [(1,1)]

root_dir = ROOT_DIR
file_names_data = []
file_names_scat = []

# process data
...

# scat transforming test data
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        try:
            print("\tscat transforming with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
            file_name_scat_test = scu.scat_transform(file_name_test, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
        except:
            print("\texception occurred with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
