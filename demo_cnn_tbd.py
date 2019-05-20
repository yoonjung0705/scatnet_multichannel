import numpy as np
import scatnet as scn
import sim_utils as siu
import scatnet_utils as scu
import pytorch_cnn as pyc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_len = 2**11
avg_len = 2**8
n_data = 1000
n_data_val = 400
n_data_test = 600

n_filter_octave = [1,1]

scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)


diff_coef_ratios = np.arange(2,10,2)
dt = 0.01
k_ratios = np.arange(1,4,1)
#k_ratios = [1]

traj_tbd = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data)
traj_tbd = traj_tbd.reshape(-1,2,traj_tbd.shape[-1])
scat_tbd = scn.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave)
S_tbd = scat.transform(traj_tbd)
S_tbd_merge = scu.merge_channels(S_tbd)
S_tbd_merge_log = scu.log_scat(S_tbd_merge)
S_tbd_merge_log_stack = scu.stack_scat(S_tbd_merge_log)
S_tbd_merge_log_stack_mean = S_tbd_merge_log_stack.mean(axis=-1)
S_tbd_merge_log_stack_mean_tensor = torch.tensor(S_tbd_merge_log_stack_mean, dtype=torch.float32)

traj_tbd_val = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data_val)
traj_tbd_val = traj_tbd_val.reshape(-1,2,traj_tbd_val.shape[-1])
scat_tbd_val = scn.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave)
S_tbd_val = scat.transform(traj_tbd_val)
S_tbd_val_merge = scu.merge_channels(S_tbd_val)
S_tbd_val_merge_log = scu.log_scat(S_tbd_val_merge)
S_tbd_val_merge_log_stack = scu.stack_scat(S_tbd_val_merge_log)
S_tbd_val_merge_log_stack_mean = S_tbd_val_merge_log_stack.mean(axis=-1)
S_tbd_val_merge_log_stack_mean_tensor = torch.tensor(S_tbd_val_merge_log_stack_mean, dtype=torch.float32)

traj_tbd_test = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data_test)
traj_tbd_test = traj_tbd_test.reshape(-1,2,traj_tbd_test.shape[-1])
scat_tbd_test = scn.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave)
S_tbd_test = scat.transform(traj_tbd_test)
S_tbd_test_merge = scu.merge_channels(S_tbd_test)
S_tbd_test_merge_log = scu.log_scat(S_tbd_test_merge)
S_tbd_test_merge_log_stack = scu.stack_scat(S_tbd_test_merge_log)
S_tbd_test_merge_log_stack_mean = S_tbd_test_merge_log_stack.mean(axis=-1)
S_tbd_test_merge_log_stack_mean_tensor = torch.tensor(S_tbd_test_merge_log_stack_mean, dtype=torch.float32)

net = pyc.Net(S_tbd_merge_log_stack_mean_tensor.shape[1],200,100,2)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
loss_arr = []
val_arr = []


target = torch.zeros(S_tbd_merge_log_stack_mean_tensor.shape[0],2)
a = -1
c = 0
for _ in range(len(k_ratios)):
    a = a + 1
    b = -1
    for _ in range(len(diff_coef_ratios)):
        b = b +1
        for _ in range(n_data):
            target[c,:] = torch.tensor([k_ratios[a],diff_coef_ratios[b]])
            c = c + 1

target_val = torch.zeros(S_tbd_val_merge_log_stack_mean_tensor.shape[0],2)
a = -1
c = 0
for _ in range(len(k_ratios)):
    a = a + 1
    b = -1
    for _ in range(len(diff_coef_ratios)):
        b = b +1
        for _ in range(n_data_val):
            target_val[c,:] = torch.tensor([k_ratios[a],diff_coef_ratios[b]])
            c = c + 1


for _ in range(1000):
      output = net(S_tbd_merge_log_stack_mean_tensor)
      validation = net(S_tbd_val_merge_log_stack_mean_tensor)
      loss = criterion(output, target)
      val = criterion(validation, target_val)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_arr.append(loss.item())
      val_arr.append(val.item())

