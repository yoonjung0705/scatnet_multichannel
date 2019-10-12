import numpy as np
import sim_utils as siu
import scat_utils as scu
import net_utils as nu
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

scat = scu.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)


diff_coef_ratios = np.arange(2,10,2)
#diff_coef_ratios = [4] 
dt = 0.01
#k_ratios = np.arange(1,4,1)
k_ratios = np.arange(2,10,2)
#k_ratios = [1]

traj_tbd = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data)
traj_tbd = traj_tbd.reshape(-1,2,traj_tbd.shape[-1])
scat_tbd = scu.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave)
S_tbd = scat.transform(traj_tbd)
######## This part is use the time information and do not the spatial channels ###############
S_tbd_log = scu.log_scat(S_tbd)
S_tbd_log_stack = scu.stack_scat(S_tbd_log)
S_tbd_log_stack = S_tbd_log_stack.reshape(S_tbd_log_stack.shape[0],S_tbd_log_stack.shape[1]*S_tbd_log_stack.shape[2]*S_tbd_log_stack.shape[3])
S_tbd_log_stack_tensor = torch.tensor(S_tbd_log_stack, dtype=torch.float32)
######## This part is average the time information and merge the spatial channels #############
# S_tbd_merge = scu.merge_channels(S_tbd)
# S_tbd_merge_log = scu.log_scat(S_tbd_merge)
# S_tbd_merge_log_stack = scu.stack_scat(S_tbd_merge_log)
# S_tbd_merge_log_stack_mean = S_tbd_merge_log_stack.mean(axis=-1)
# S_tbd_merge_log_stack_mean_tensor = torch.tensor(S_tbd_merge_log_stack_mean, dtype=torch.float32)

traj_tbd_val = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data_val)
traj_tbd_val = traj_tbd_val.reshape(-1,2,traj_tbd_val.shape[-1])
scat_tbd_val = scu.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave)
S_tbd_val = scat.transform(traj_tbd_val)
######## This part is use the time information and do not the spatial channels ###############
S_tbd_val_log = scu.log_scat(S_tbd_val)
S_tbd_val_log_stack = scu.stack_scat(S_tbd_val_log)
S_tbd_val_log_stack = S_tbd_val_log_stack.reshape(S_tbd_val_log_stack.shape[0],S_tbd_val_log_stack.shape[1]*S_tbd_val_log_stack.shape[2]*S_tbd_val_log_stack.shape[3])
S_tbd_val_log_stack_tensor = torch.tensor(S_tbd_val_log_stack, dtype=torch.float32)
######## This part is average the time information and merge the spatial channels #############
# S_tbd_val_merge = scu.merge_channels(S_tbd_val)
# S_tbd_val_merge_log = scu.log_scat(S_tbd_val_merge)
# S_tbd_val_merge_log_stack = scu.stack_scat(S_tbd_val_merge_log)
# S_tbd_val_merge_log_stack_mean = S_tbd_val_merge_log_stack.mean(axis=-1)
# S_tbd_val_merge_log_stack_mean_tensor = torch.tensor(S_tbd_val_merge_log_stack_mean, dtype=torch.float32)

traj_tbd_test = siu.sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=n_data_test)
traj_tbd_test = traj_tbd_test.reshape(-1,2,traj_tbd_test.shape[-1])
scat_tbd_test = scu.ScatNet(data_len, avg_len, n_filter_octave = n_filter_octave)
S_tbd_test = scat.transform(traj_tbd_test)
######## This part is use the time information and do not the spatial channels ###############
S_tbd_test_log = scu.log_scat(S_tbd_test)
S_tbd_test_log_stack = scu.stack_scat(S_tbd_test_log)
S_tbd_test_log_stack = S_tbd_test_log_stack.reshape(S_tbd_test_log_stack.shape[0],S_tbd_test_log_stack.shape[1]*S_tbd_test_log_stack.shape[2]*S_tbd_test_log_stack.shape[3])
S_tbd_test_log_stack_tensor = torch.tensor(S_tbd_test_log_stack, dtype=torch.float32)
######## This part is average the time information and merge the spatial channels #############
# S_tbd_test_merge = scu.merge_channels(S_tbd_test)
# S_tbd_test_merge_log = scu.log_scat(S_tbd_test_merge)
# S_tbd_test_merge_log_stack = scu.stack_scat(S_tbd_test_merge_log)
# S_tbd_test_merge_log_stack_mean = S_tbd_test_merge_log_stack.mean(axis=-1)
# S_tbd_test_merge_log_stack_mean_tensor = torch.tensor(S_tbd_test_merge_log_stack_mean, dtype=torch.float32)

net_k = nu.Net(S_tbd_log_stack_tensor.shape[1],200,100,1)
net_T = nu.Net(S_tbd_log_stack_tensor.shape[1],200,100,1)
optimizer_k = optim.SGD(net_k.parameters(), lr=0.01)
optimizer_T = optim.SGD(net_T.parameters(), lr=0.01)
criterion = nn.MSELoss()
loss_arr_k = []
loss_arr_T = []
val_arr_k = []
val_arr_T = []

target_k_ratio = torch.zeros(S_tbd_log_stack_tensor.shape[0],1)
c = 0
for i in range(len(k_ratios)):
    for _ in range(len(diff_coef_ratios)):
        for _ in range(n_data):
            target_k_ratio[c] = torch.tensor(k_ratios[i], dtype=torch.float32)
            c = c + 1

target_T_ratio = torch.zeros(S_tbd_log_stack_tensor.shape[0],1)
c = 0
for _ in range(len(k_ratios)):
     for j in range(len(diff_coef_ratios)):
         for _ in range(n_data):
            target_T_ratio[c] = torch.tensor(diff_coef_ratios[j], dtype=torch.float32)
            c = c + 1

target_k_ratio_val = torch.zeros(S_tbd_val_log_stack_tensor.shape[0],1)
c = 0
for _ in range(len(k_ratios)):
     for j in range(len(diff_coef_ratios)):
          for _ in range(n_data_val):
            target_k_ratio_val[c] = torch.tensor(diff_coef_ratios[j])
            c = c + 1


target_T_ratio_val = torch.zeros(S_tbd_val_log_stack_tensor.shape[0],1)
c = 0
for _ in range(len(k_ratios)):
     for j in range(len(diff_coef_ratios)):
          for _ in range(n_data_val):
            target_T_ratio_val[c] = torch.tensor(diff_coef_ratios[j], dtype=torch.float32)
            c = c + 1


for i in range(1000):
      output_k = net_k(S_tbd_log_stack_tensor)
      validation_k = net_k(S_tbd_val_log_stack_tensor)
      loss_k = criterion(output_k, target_k_ratio)
      val_k = criterion(validation_k, target_k_ratio_val)
      optimizer_k.zero_grad()
      loss_k.backward()
      optimizer_k.step()
      loss_arr_k.append(loss_k.item())
      val_arr_k.append(val_k.item())
      print(i)

for i in range(1000):
      output_T = net_T(S_tbd_log_stack_tensor)
      validation_T = net_T(S_tbd_val_log_stack_tensor)
      loss_T = criterion(output_T, target_T_ratio)
      val_T = criterion(validation_T, target_T_ratio_val)
      optimizer_T.zero_grad()
      loss_T.backward()
      optimizer_T.step()
      loss_arr_T.append(loss_T.item())
      val_arr_T.append(val_T.item())
      print(i)
