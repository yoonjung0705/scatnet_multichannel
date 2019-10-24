'''this is a module that demonstrates the Lipschitz continuous property of the Invariant Scattering transform'''

'''standard imports'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

'''custom libraries'''
import scat_utils as scu
import sim_utils as siu

#plt.style.use('dark_background')
fontsize_title = 18
fontsize_label = 14
data_len = 2**12
avg_len = 2**8
disp_len = 2**7
dt = 0.001
n_data = 10
eps = 0.015

padded_len = np.ceil(data_len * (1 + eps * n_data))
t = np.arange(0, data_len) * dt
t_padded = np.arange(0, padded_len) * dt
a_padded = np.sin(10 * t_padded) + np.sin(30 * t_padded) + np.sin(50 * t_padded)

f = interpolate.interp1d(t_padded, a_padded)
scat = scu.ScatNet(data_len=data_len, avg_len=avg_len)

data = [f(t / (1 + eps * idx)) for idx in range(n_data)]
data = np.stack(data, axis=0) # shaped (n_data, data_len)

fig1, ax1 = plt.subplots()

ax1.plot(data[:, :disp_len].swapaxes(0,1))
ax1.set_title('Dilated time series', fontsize=fontsize_title)
ax1.set_xlabel('Time', fontsize=fontsize_label)
ax1.set_ylabel('Amplitude', fontsize=fontsize_label)

dilation = 1 + eps * np.arange(n_data)
S = scat.transform(data[:, np.newaxis, :])
S = scu.stack_scat(S) # (n_data, 1, n_nodes, data_scat_len)

# subtract undilated signal's scat transform result for each scat transform result using broadcasting
diff_scat = S - S[0:1]
diff_data = data - data[0:1]
diff_fourier = np.fft.fft(diff_data, axis=-1) # since fft is a linear operator, Uf1 - Uf2 = U(f1 - f2)

# compute the distance which is the sum of the absolute values of the difference, followed by the square root
dist_scat = np.sqrt((np.abs(diff_scat)**2).reshape([n_data, -1]).sum(axis=-1))
dist_data = np.sqrt((np.abs(diff_data)**2).reshape([n_data, -1]).sum(axis=-1))
dist_fourier = np.sqrt((np.abs(diff_fourier)**2).reshape([n_data, -1]).sum(axis=-1))

fig2, ax2 = plt.subplots()
dist_scat_rescaled = dist_scat / dist_scat[1] 
dist_data_rescaled = dist_data / dist_data[1] 
dist_fourier_rescaled = dist_fourier / dist_fourier[1] 

ax2.plot(dilation, dist_scat_rescaled, label='ScatNet distance')
ax2.plot(dilation, dist_data_rescaled, label='L2 distance')
ax2.plot(dilation, dist_fourier_rescaled, label='Fourier L2 distance')

dashed_line = np.arange(n_data)
ax2.plot(dilation, dashed_line, '--')
ax2.legend()

ax2.set_title('Feature space distance', fontsize=fontsize_title)
ax2.set_xlabel('Dilation', fontsize=fontsize_label)
ax2.set_ylabel('Distance', fontsize=fontsize_label)
plt.show()
