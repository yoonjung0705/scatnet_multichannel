'''module for performing invariant scattering transformation on multichannel time series'''
import os
import numpy as np

def T_to_J(T, filt_opt):
    filt_opt = filt_opt.copy() # if you don't do this, the input argument filt_opt will get updated and have all these B and phi_bw_multiplier keys!
    filt_opt = fill_struct(filt_opt, Q=1)
    filt_opt = fill_struct(filt_opt, B=filt_opt['Q'])
    Q = filt_opt['Q']
    B = filt_opt['B']
    if isinstance(Q, list):
        bw_mult = list(1 + (np.array(Q)==1).astype('int'))
    else:
        bw_mult = 1 + int(np.array(Q)==1)
    filt_opt = fill_struct(filt_opt, phi_bw_multiplier=bw_mult)
    bw_mult = filt_opt['phi_bw_multiplier']
    if isinstance(Q, list) and isinstance(B, list):
        J = list(1 + np.round(np.log2(T / (4 * np.array(B) \
        / np.array(bw_mult))) * np.array(Q)).astype(int)) 
        # NOTE: indices should be type int
    elif Q > 0 and B > 0:
        J = 1 + int(np.round(np.log2(T / (4 * B / bw_mult)) * Q))
    
    return J

def default_filter_opt(filter_type, averaging_size):
    s = {}
    if filter_type == 'audio':
        s['Q'] = [8, 1]
        s['J'] = T_to_J(averaging_size, s)
    elif filter_type == 'dyadic':
        s['Q'] = 1
        s['J'] = T_to_J(averaging_size, s)
    else:
        raise ValueError("Invalid filter_type: should be either audio or dyadic")
    return s

def fill_struct(s, **kwargs): 
# in the matlab implementation, this function is called only with a single key-value pair, 
# such as fill_struct(s, 'precision', 'double')
    for key, value in kwargs.items():
        if key in s:
            if s[key] is None:
                s[key] = value
        else:
            s[key] = value
    return s

def morlet_freq_1d(filt_opt):
    '''
    inputs:
    ------- 
    - filt_opt: type dict

    outputs:
    -------- 
    - xi_psi: sized (J+P, ). logarithmically spaced J elements, linearly spaced P elements
    - bw_psi: sized (J+P+1, ). logarithmically spaced J elements, linearly spaced P+1 elements
    both type nparray during calculations, convertable to list at final output
    - bw_phi: float
    
    increasing index corresponds to filters with decreasing center frequency
    filters with high freq are logarithmically spaced, low freq interval is covered linearly
    
    FIXME: consider defining local variables for the key-value pairs in filt_opt during calculations
    FIXME: consider converting outputs to list type
    '''
    sigma0 = 2 / np.sqrt(3)
    
    # Calculate logarithmically spaced, band-pass filters.
    xi_psi = filt_opt['xi_psi'] * 2**(np.arange(0,-filt_opt['J'],-1) / filt_opt['Q'])
    sigma_psi = filt_opt['sigma_psi'] * 2**(np.arange(filt_opt['J']) / filt_opt['Q'])

    # Calculate linearly spaced band-pass filters so that they evenly
    # cover the remaining part of the spectrum
    step = pi * 2**(-filt_opt['J'] / filt_opt['Q']) * (1 - 1/4 * sigma0 / filt_opt['sigma_phi'] \
        * 2**( 1 / filt_opt['Q'] ) ) / filt_opt['P']
    # xi_psi = np.array(xi_psi)
    # xi_psi[filt_opt['J']:filt_opt['J']+filt_opt['P']] = filt_opt['xi_psi'] * 2**((-filt_opt['J']+1) / filt_opt['Q']) - step * np.arange(1, filt_opt['P'] + 1)
    xi_psi_linspace = filt_opt['xi_psi'] * 2**((-filt_opt['J']+1) / filt_opt['Q']) \
    - step * np.arange(1, filt_opt['P'] + 1)
    xi_psi = np.concatenate([xi_psi, xi_psi_linspace], axis=0) 
    # sigma_psi = np.array(sigma_psi)
    # sigma_psi[filt_opt['J']:filt_opt['J']+1+filt_opt['P']] = filt_opt['sigma_psi'] * 2**((filt_opt['J'] - 1) / filt_opt['Q'])
    sigma_psi_linspace = np.tile(filt_opt['sigma_psi'] * 2**((filt_opt['J'] - 1) / filt_opt['Q']), 
        (1, 1+filt_opt['P']))
    sigma_psi = np.concatenate([sigma_psi, sigma_psi_linspace], axis=0)
    
    # Calculate band-pass filter
    sigma_phi = filt_opt['sigma_phi'] * 2**((filt_opt['J']-1) / filt_opt['Q'])

    # Convert (spatial) sigmas to (frequential) bandwidths
    bw_psi = pi / 2 * sigma0 / sigma_psi
    if not filt_opt['phi_dirac']:
        bw_phi = pi / 2 * sigma0 / sigma_phi
    else:
        bw_phi = 2 * pi

    return (xi_psi, bw_psi, bw_phi)

def optimize_filter(filter_f, lowpass, options):
    options = fill_struct(options, truncate_threshold=1e-3);
    options = fill_struct(options, filter_format=fourier_multires);

    if options.filter_format == 'fourier'):
        filt = filter_f
    elif (options.filter_format == 'fourier_multires'):
        filt = periodize_filter(filter_f)
    elif options.filter_format == 'fourier_truncated'):
        filt = truncate_filter(filter_f,options.truncate_threshold,lowpass)
    else:
        raise ValueError('Unknown filter format {}'.format(options.filter_format))
    return filt


