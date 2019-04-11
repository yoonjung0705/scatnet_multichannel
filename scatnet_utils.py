import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scatnet as sn

def dist_scat(x1, x2):
    '''computes distance between signals after scattering transform'''
    sum_sqs = 0
    for m in range(len(x1)):
        n_signals = len(x1[m]['signal'])
        for n in range(n_signals):
            tmp = np.sum((x1[m]['signal'][n] - x2[m]['signal'][n])**2)
            sum_sqs += tmp
            
    return np.sqrt(sum_sqs)
            
def dist_l2(x1, x2):
    '''computes euclidean distance between signals'''
    sum_sqs = np.sum((x1 - x2)**2)

    return np.sqrt(sum_sqs)

def dist_fourier_modulus(x1, x2):
    '''computes distance between signals' fourier modulus'''
    x1_f = np.abs(np.fft.fft(x1))
    x2_f = np.abs(np.fft.fft(x2))

    sum_sqs = np.sum(np.abs(x1_f - x2_f)**2)

    return np.sqrt(sum_sqs)

def dilate(data, scale):
    '''dilates 1d signal upto the given scale'''
    assert(scale > 1), "Invalid dilation amount. scale should be larger than 1"
    x0 = np.arange(len(data))
    fit = interpolate.interp1d(x0, data)
    x_new = x0 / scale
    data_new = fit(x_new)

    return data_new
