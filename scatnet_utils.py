import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
import scatnet as sn

def dist_scat(x1, x2):
    '''computes distance between signals after scattering transform'''
    sum_sqs = 0
    for m in range(len(x1)):
        n_signals = len(x1[m]['signal'])
        for n in range(n_signals):
            tmp = np.sum((x1[m]['signal'][n] - x2[m]['signal'][n])**2, axis=-1) 
            # FIXME: currently, axis=-1 works well for both input data shaped (n_data, data_len) and (data_len,). 
            # Later when allowing multiple channels, this might break
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

def stack_scat(x):
    '''
    reshapes scattering transform signal into ndarray
    
    inputs:
    -------
    x: list type object resulting from scat.transform() 
    
    outputs:
    --------
    x_stack: ndarray shaped(n_data, n_channels, n_nodes, scat_transform_single_node_len)
    '''
    x_stack = []
    for m in range(len(x)):
        x_stack_m = np.stack(x[m]['signal'], axis=-2)
        x_stack.append(x_stack_m)
    x_stack = np.concatenate(x_stack, axis=-2)

    return x_stack

def log_scat(x, eps=1.e-6):
    '''
    returns logarithm of scattering transform's absolute values
    
    inputs:
    -------
    x: list type object resulting from scat.transform()
    eps: small amount added to prevent the log results blowing up
    
    outputs:
    --------
    x: instance of scat.transform() whose values are the logarithm of the absolute values of the input
    '''
    x = copy.deepcopy(x)
    for m in range(len(x)):
        n_signals = len(x[m]['signal'])
        for n in range(n_signals):
            res = x[m]['meta']['resolution'][n]
            x[m]['signal'][n] = np.log(np.abs(x[m]['signal'][n]) + eps * 2**(res/2))
    return x

def scat_features(x, params, avg_len, n_filter_octave=[1, 1]):
    '''returns feature matrix X from a set of time series using the scattering transform
    calculates the logarithm of the scattering transform and takes the mean along the time axis
    for the filter format and the boundary conditions, default parameters are used
    (filter_format='fourier_truncated', mode='symm')

    inputs:
    -------
    x: rank d array whose last dimension corresponds to the time axis
    params: list of parameters. parameters can be either lists or 1d arrays
    avg_len: scaling function width for scattering transform
    n_filter_octave: number of filters per octave for scattering transform
    
    outputs:
    --------
    X: rank 2 array sized (n_timeseries, n_nodes)
    y: rank 1 array sized (n_timeseries,) denoting the simulation parameters

    '''
    data_len = x.shape[-1]
    for idx, s in enumerate(x.shape[:-1]):
        assert(s == len(params[idx])), "Array shape does not comply with number of parameters"
    scat = scn.ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    S = scat.transform(x)
    S_log = scu.log_scat(S)
    S_log = scu.stack_scat(S_log)
    S_log_mean = S_log.mean(axis=-1)
    S_log_mean = np.reshape(S_log_mean, (-1, S_log_mean.shape[-1]))
    
    return S_log_mean

def merge_channels(x):
    '''returns an instance of the result of scat.transform() whose channels are merged into a single channel

    inputs:
    -------
    x: list type instance resulting from scat.transform()
    
    outputs:
    --------
    x: instance of scat.transform() whose values are replaced with the l2 norm along the channel axis
    '''
    x = copy.deepcopy(x)
    for m in range(len(x)):
        n_signals = len(x[m]['signal'])
        for n in range(n_signals):
            x[m]['signal'][n] = np.sqrt(np.sum(np.abs(x[m]['signal'][n])**2, axis=1))
    return x

