import numpy as np 
import scatnet as scn 
import sim_utils as siu 
import scatnet_utils as scu 
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

scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
# precision for external parameters                                            
# simulate brownian                                                            
diff_coefs_brw = np.arange(4,8,1)
dt = 0.01
traj_brw = siu.sim_brownian(data_len, diff_coefs_brw, dt=dt, n_data=n_data)
scat_brw = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
traj_brw = traj_brw.reshape(-1, 1, traj_brw.shape[-1])
S_brw = scat.transform(traj_brw)
S_brw_log = scu.log_scat(S_brw)
S_brw_log = scu.stack_scat(S_brw_log)
S_brw_log_mean = S_brw_log.mean(axis=-1)
S_brw_log_mean = np.reshape(S_brw_log_mean, (-1, S_brw_log_mean.shape[-1]))
S_brw_log_mean_tensor =torch.tensor(S_brw_log_mean, dtype=torch.float32)

traj_brw_val = siu.sim_brownian(data_len, diff_coefs_brw, dt=dt, n_data=n_data_val)
scat_brw_val = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
traj_brw_val = traj_brw_val.reshape(-1, 1, traj_brw_val.shape[-1])
S_brw_val = scat.transform(traj_brw_val)
S_brw_val_log = scu.log_scat(S_brw_val)
S_brw_val_log = scu.stack_scat(S_brw_val_log)
S_brw_val_log_mean = S_brw_val_log.mean(axis=-1)
S_brw_val_log_mean = np.reshape(S_brw_val_log_mean, (-1, S_brw_val_log_mean.shape[-1]))
S_brw_val_log_mean_tensor =torch.tensor(S_brw_val_log_mean, dtype=torch.float32)

traj_brw_test = siu.sim_brownian(data_len, diff_coefs_brw, dt=dt, n_data=n_data_test)
scat_brw_test = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
traj_brw_test = traj_brw_test.reshape(-1, 1, traj_brw_test.shape[-1])
S_brw_test = scat.transform(traj_brw_test)
S_brw_test_log = scu.log_scat(S_brw_test)
S_brw_test_log = scu.stack_scat(S_brw_test_log)
S_brw_test_log_mean = S_brw_test_log.mean(axis=-1)
S_brw_test_log_mean = np.reshape(S_brw_test_log_mean, (-1, S_brw_test_log_mean.shape[-1]))
S_brw_test_log_mean_tensor =torch.tensor(S_brw_test_log_mean, dtype=torch.float32)

net = nu.Net(S_brw_log_mean_tensor.shape[1],100,50,1)
optimizer = optim.SGD(net.parameters(), lr=0.01) 
criterion = nn.MSELoss() 
loss_arr = [] 
val_arr = []

target = torch.zeros(S_brw_log_mean_tensor.shape[0],1) 
a = 0 
b = 0 
for _ in range(4): 
      b = b + 1 
      for _ in range(n_data): 
          target[a] = torch.tensor(b) 
          a = a + 1   

target_val = torch.zeros(S_brw_val_log_mean_tensor.shape[0],1)
a = 0
b = 0
for _ in range(4):
      b = b + 1
      for _ in range(n_data_val):
          target_val[a] = torch.tensor(b)
          a = a + 1


for _ in range(1000): 
      output = net(S_brw_log_mean_tensor)
      validation = net(S_brw_val_log_mean_tensor) 
      loss = criterion(output, target)
      val = criterion(validation, target_val) 
      optimizer.zero_grad() 
      loss.backward() 
      optimizer.step() 
      loss_arr.append(loss.item()) 
      val_arr.append(val.item())

