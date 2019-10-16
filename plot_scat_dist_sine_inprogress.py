import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

import scat_utils as scu

plt.style.use('dark_background')
fontsize_title = 18
fontsize_labels = 18
#padded_len = 2**17
data_len = 2**8
avg_len = 2**5
n_data = 8
eps = 0.015
padded_len = data_len * (1 + eps * n_data)

x = np.arange(0, data_len)
a = np.sin(np.pi * x) + np.sin(3 * np.pi * x) + np.sin(5 * np.pi * x)

f = interpolate.interp1d(x, a)
scat = scu.ScatNet(data_len=data_len, avg_len=avg_len)

data = [f(x * (1-eps*idx)) for idx in range(n_data)]
data = np.stack(data, axis=0) # shaped (n_data, data_len)
#x2 = x * (1-eps*1)
#x3 = x * (1-eps*2)
#x4 = x * (1-eps*3)
#x5 = x * (1-eps*4)
#x6 = x * (1-eps*5)
#x7 = x * (1-eps*6)
#x8 = x * (1-eps*7)

scales = 1 - eps * np.arange(n_data)
#a2 = f(x2)
#a3 = f(x3)
#a4 = f(x4)
#a5 = f(x5)
#a6 = f(x6)
#a7 = f(x7)
#a8 = f(x8)

fig1, ax1 = plt.subplots()

ax1.plot(data.swapaxes(0,1))
ax1.set_title('Dilated time series', fontsize=fontsize_title)
ax1.set_xlabel('x', fontsize=fontsize_labels)
ax1.set_ylabel('Amplitude', fontsize=fontsize_labels)

#scales_rev = scales[::-1]
#dilation = scales_rev / scales_rev[0]

#a = a[:data_len]
#a2 = a2[:data_len]
#a3 = a3[:data_len]
#a4 = a4[:data_len]
#a5 = a5[:data_len]
#a6 = a6[:data_len]
#a7 = a7[:data_len]
#a8 = a8[:data_len]

S = scat.transform(data[:, np.newaxis, :])
S = scu.stack_scat(S) # (n_data, 1, n_nodes, data_scat_len)
#y2 = scat.transform(a2[np.newaxis, np.newaxis, :])
#y3 = scat.transform(a3[np.newaxis, np.newaxis, :])
#y4 = scat.transform(a4[np.newaxis, np.newaxis, :])
#y5 = scat.transform(a5[np.newaxis, np.newaxis, :])
#y6 = scat.transform(a6[np.newaxis, np.newaxis, :])
#y7 = scat.transform(a7[np.newaxis, np.newaxis, :])
#y8 = scat.transform(a8[np.newaxis, np.newaxis, :])


def diff_scat(S1,S2):
    sum_sqs = 0
    for m in range(scat.n_layers + 1):
        n_signals = len(S1[m]['signal'])
        for n in range(n_signals):
#             print(m)
#             print(n)
#             print(S1[m]['signal'][n])
            tmp = np.sum( (  S1[m]['signal'][n] - S2[m]['signal'][n]  )**2 )
#             print(tmp)
            sum_sqs += tmp
    return np.sqrt(sum_sqs)

def diff_sig(S1,S2):
    sum_sqs = np.sum( (S1 - S2)**2 )

    return np.sqrt(sum_sqs)

def diff_fourier_modulus(S1,S2):
    af1 = np.abs(np.fft.fft(S1))
    af2 = np.abs(np.fft.fft(S2))
    sum_sqs = np.sum( np.abs(af1 - af2)**2 )

    return np.sqrt(sum_sqs)

diff_scat88 = diff_scat(y8, y8)
diff_scat87 = diff_scat(y8, y7)
diff_scat86 = diff_scat(y8, y6)
diff_scat85 = diff_scat(y8, y5)
diff_scat84 = diff_scat(y8, y4)
diff_scat83 = diff_scat(y8, y3)
diff_scat82 = diff_scat(y8, y2)
diff_scat81 = diff_scat(y8, y)

diff_sig88 = diff_sig(a8, a8)
diff_sig87 = diff_sig(a8, a7)
diff_sig86 = diff_sig(a8, a6)
diff_sig85 = diff_sig(a8, a5)
diff_sig84 = diff_sig(a8, a4)
diff_sig83 = diff_sig(a8, a3)
diff_sig82 = diff_sig(a8, a2)
diff_sig81 = diff_sig(a8, a)

diff_fourier88 = diff_fourier_modulus(a8, a8)
diff_fourier87 = diff_fourier_modulus(a8, a7)
diff_fourier86 = diff_fourier_modulus(a8, a6)
diff_fourier85 = diff_fourier_modulus(a8, a5)
diff_fourier84 = diff_fourier_modulus(a8, a4)
diff_fourier83 = diff_fourier_modulus(a8, a3)
diff_fourier82 = diff_fourier_modulus(a8, a2)
diff_fourier81 = diff_fourier_modulus(a8, a)

fig2, ax2 = plt.subplots()
diff_scat_arr = np.array([diff_scat88, diff_scat87, diff_scat86, diff_scat85, diff_scat84, diff_scat83, diff_scat82, diff_scat81])
# ax2.plot(dilation, diff_scat_arr, label='Scattering')

diff_sig_arr = np.array([diff_sig88, diff_sig87, diff_sig86, diff_sig85, diff_sig84, diff_sig83, diff_sig82, diff_sig81])
diff_sig_arr_rescaled = diff_sig_arr * diff_scat_arr[1] / diff_sig_arr[1]

ax2.plot(dilation, diff_sig_arr_rescaled, label='L2 rescaled')

diff_fourier_arr = np.array([diff_fourier88, diff_fourier87, diff_fourier86, diff_fourier85, diff_fourier84, diff_fourier83, diff_fourier82, diff_fourier81])
diff_fourier_arr_rescaled = diff_fourier_arr * diff_scat_arr[1] / diff_fourier_arr[1]
# diff_fourier_arr_rescaled_multiplied = diff_fourier_arr_rescaled * 0.8
ax2.plot(dilation, diff_fourier_arr_rescaled, label='Fourier modulus L2 rescaled')

dotted_line = (diff_scat_arr[1] - diff_scat_arr[0]) / (dilation[1] - dilation[0]) * (dilation - dilation[0]) + diff_scat_arr[0]
ax2.plot(dilation, dotted_line, '--')
ax2.legend()

ax2.set_title('Signal distance', fontsize=18)
ax2.set_xlabel('dilation', fontsize=18)
ax2.set_yticks([])

plt.show()
