'''module for performing invariant scattering transformation on multichannel time series'''
import os
import numpy as np

'''
FIXME: consider name change for filter_options, filter_type
FIXME: for functions that allow signal to be rank 1 array for number of data being 1, only allow rank 2 inputs
(even for 1 signal, it should be shaped (1, data_len)). This is important as we want to add the channel dimension as well,
although calculating and combining different channels might be possible to be done at a higher level (such as in function scat(), etc)
'''

def T_to_J(T, filt_opt):
    '''
    calculates maximal wavelet scale J
    
    inputs:
    -------
    - T: int type, length of signal in units of samples
    - filt_opt: dict type object with parameters specifying a filter

    outputs:
    --------
    - J: int type ot=r list type with elements int type. maximal wavelet scale

    FIXME: consider name change of Q, B, J, filt_opt
    '''
    filt_opt = filt_opt.copy() # prevents filt_opt change upon function call
    filt_opt = fill_struct(filt_opt, Q=1)
    filt_opt = fill_struct(filt_opt, B=filt_opt['Q'])
    Q = filt_opt['Q']
    B = filt_opt['B']
    if isinstance(Q, list):
        bw_mult = list(1 + (np.array(Q)==1).astype('int')) # for Q list's values that are equal to 1, bw_mult's element is 2. for values that are not, bm_mult's element is 1
    else:
        bw_mult = 1 + int(np.array(Q)==1)
    filt_opt = fill_struct(filt_opt, phi_bw_multiplier=bw_mult)
    # get bw_mult from filt_opt. The reason not using bw_mult right away is because filt_opt might have alreayd had phi_bw_multiplier as a valid value
    bw_mult = filt_opt['phi_bw_multiplier']
    if isinstance(Q, list) and isinstance(B, list):
        J = list(1 + np.round(np.log2(T / (4 * np.array(B) \
        / np.array(bw_mult))) * np.array(Q)).astype(int)) 
        # NOTE: indices should be type int
    elif Q > 0 and B > 0:
        J = 1 + int(np.round(np.log2(T / (4 * B / bw_mult)) * Q))
    else:
        raise ValueError("Invalid type of Q or B: must be list or numeric")
    
    return J

def default_filter_opt(filter_type, avg_len):
    '''
    returns dict type object containing default parameters Q, J for filters
    inputs:
    -------
    - filter_type: "audio", "dyadic"
    - avg_len: int type number representing width of scaling function in units of samples

    outputs:
    --------
    - s: dict type object containing default parameters for filters

    FIXME: change variable name s
    '''
    s = {}
    if filter_type == 'audio':
        s['Q'] = [8, 1]
        s['J'] = T_to_J(avg_len, s)
    elif filter_type == 'dyadic':
        s['Q'] = 1
        s['J'] = T_to_J(avg_len, s)
    else:
        raise ValueError("Invalid filter_type: should be either audio or dyadic")
    return s

def fill_struct(s, **kwargs):
    '''
    for a given dictionary and a number of key-value pairs, fills in the key-values of the
    dictionary if the key does not exist or if the value of the key is empty

    inputs:
    -------
    - s: dict type that may or may not contain the keys given by user

    outputs:
    --------
    - s: type dict object that is updated with the given key-value pairs. For keys that
    originally did not exist, the key-value pair is updated. For keys that originally existed
    are updated only if the values were None
    
    FIXME: consider name change of s. Consider replacing function with more readable function''' 
    for key, value in kwargs.items():
        if key in s:
            if s[key] is None:
                s[key] = value
        else:
            s[key] = value
    return s

def morlet_freq_1d(filt_opt):
    '''
    given filter options, returns parameters xi, bw of filter banks
    inputs:
    ------- 
    - filt_opt: type dict with the following keys:
    xi_psi, sigma_psi, sigma_phi, J, Q, P: all numeric
    phi_dirac: type bool
    As all values in filt_opt dict are scalars, filt_opt argument does not change upon function call

    outputs:
    -------- 
    - xi_psi: list sized (J+P, ). logarithmically spaced J elements, linearly spaced P elements
    - bw_psi: list sized (J+P+1, ). logarithmically spaced J elements, linearly spaced P+1 elements
    both type nparray during calculations, converted to list at final output
    - bw_phi: float
    
    increasing index corresponds to filters with decreasing center frequency
    filters with high freq are logarithmically spaced, low freq interval is covered linearly
    NOTE: xi_psi can in theory have negative center frequencies in this function
    
    FIXME: consider defining local variables for the key-value pairs in filt_opt during calculations
    FIXME: consider not converting outputs to list type
    REVIEW: manually compare results with MATLAB version
    '''
    sigma0 = 2 / np.sqrt(3)
    
    # calculate logarithmically spaced band-pass filters.
    xi_psi = filt_opt['xi_psi'] * 2**(np.arange(0,-filt_opt['J'],-1) / filt_opt['Q'])
    sigma_psi = filt_opt['sigma_psi'] * 2**(np.arange(filt_opt['J']) / filt_opt['Q'])
    # calculate linearly spaced band-pass filters so that they evenly
    # cover the remaining part of the spectrum
    step = np.pi * 2**(-filt_opt['J'] / filt_opt['Q']) * (1 - 1/4 * sigma0 / filt_opt['sigma_phi'] \
        * 2**( 1 / filt_opt['Q'] ) ) / filt_opt['P']
    # xi_psi = np.array(xi_psi)
    # xi_psi[filt_opt['J']:filt_opt['J']+filt_opt['P']] = filt_opt['xi_psi'] * 2**((-filt_opt['J']+1) / filt_opt['Q']) - step * np.arange(1, filt_opt['P'] + 1)
    xi_psi_lin = filt_opt['xi_psi'] * 2**((-filt_opt['J']+1) / filt_opt['Q']) \
    - step * np.arange(1, filt_opt['P'] + 1)
    xi_psi = np.concatenate([xi_psi, xi_psi_lin], axis=0) 
    # sigma_psi = np.array(sigma_psi)
    # sigma_psi[filt_opt['J']:filt_opt['J']+1+filt_opt['P']] = filt_opt['sigma_psi'] * 2**((filt_opt['J'] - 1) / filt_opt['Q'])
    sigma_psi_lin = np.full((1+filt_opt['P'],), fill_value=filt_opt['sigma_psi'] \
        * 2**((filt_opt['J'] - 1) / filt_opt['Q']))
    sigma_psi = np.concatenate([sigma_psi, sigma_psi_lin], axis=0)
    # calculate band-pass filter
    sigma_phi = filt_opt['sigma_phi'] * 2**((filt_opt['J']-1) / filt_opt['Q'])
    # convert (spatial) sigmas to (frequential) bandwidths
    bw_psi = np.pi / 2 * sigma0 / sigma_psi
    if not filt_opt['phi_dirac']:
        bw_phi = np.pi / 2 * sigma0 / sigma_phi
    else:
        bw_phi = 2 * np.pi
    return list(xi_psi), list(bw_psi), bw_phi

def optimize_filter(filter_f, lowpass, options):
    '''
    inputs:
    -------
    - filter_f: rank 1 list or nparray type. filter in the frequency domain (indicated by _f)
    - lowpass: parameter required for fourier_truncated case. FIXME: add details later
    - options: dict type containing value 'filter_format'. this indicates how filter_f will be modified

    outputs:
    --------
    - filt: np.array type or list (if fourier_multires) type. If list type, the elements are nparrays at different resolutions

    FIXME: I should add truncate_filter() as this function seems to be used for default settings
    '''
    options = fill_struct(options, truncate_threshold=1e-3)
    options = fill_struct(options, filter_format='fourier_multires')

    if options['filter_format'] == 'fourier':
        filt = np.array(filter_f) # converting to nparray since multires case return type has to be np.array
    elif options['filter_format'] == 'fourier_multires':
        filt = periodize_filter(filter_f)
    elif options['filter_format'] == 'fourier_truncated':
        filt = truncate_filter(filter_f, options['truncate_threshold'], lowpass)
    else:
        raise ValueError('Unknown filter format {}'.format(options['filter_format']))
    return filt

def filter_freq(filter_options):
    '''
    returns psi and phi parameters given the filter_options, which includes parameters for generating the filter and the filter_type
    FIXME: consider deprecating this function by only allowing filter type to be morlet_1d or gabor_1d...?
    FIXME: what is the difference between filter_type being morlet_1d and gabor_1d?
    
    inputs:
    -------
    - filter_options: dict type object containing filter_type and other parameters for generating the filter parameters at different resolutions

    outputs:
    --------
    - xi_psi: list sized (J+P, ). logarithmically spaced J elements, linearly spaced P elements
    - bw_psi: list sized (J+P+1, ). logarithmically spaced J elements, linearly spaced P+1 elements
    both type nparray during calculations, converted to list at final output
    - bw_phi: float
    '''
    if (filter_options['filter_type'] == 'spline_1d') or (filter_options['filter_type'] == 'selesnick_1d'):
        psi_xi, psi_bw, phi_bw = dyadic_freq_1d(filter_options)
    elif (filter_options['filter_type'] == 'morlet_1d') or (filter_options['filter_type'] == 'gabor_1d'):
        psi_xi, psi_bw, phi_bw = morlet_freq_1d(filter_options)
    else:
        raise ValueError('Unknown filter type {}'.format(filter_options['filter_type']))
    return psi_xi, psi_bw, phi_bw

def map_meta(from_meta, from_ind, to_meta, to_ind, exclude=[]):
    '''
    FIXME:this function will be deprecated (how to modify not planned yet).
    for all key-value pairs in from_meta, the columns are copied into the to_meta's key-value pairs
    including key value pairs not existing in to_meta while excluding the list of key value pairs in
    the argument "exclude". If the number of indices differ, the columns of from_ind are tiled
    to match the number of columns of to_ind
    
    inputs:
    -------
    - from_meta: dict type object
    - from_ind: list type containing indices
    - to_meta: dict type object
    - to_ind: list type containing indices
    - exclude: list type object containing keys that should not be considered when copying columns 
    NOTE: for arguments from_ind and to_ind, if the input is a single index, a scalar is allowed which will be cast to a length 1 list in the function

    outputs:
    --------
    - to_meta: dict type object which is a modification of the input argument to_meta

    NOTE: assumed that from_meta's values are valid 2d lists or rank 2 np arrays
    NOTE: MATLAB version can run with less input arguments
    FIXME: if to_ind has 4 indices and from_ind has 2 indices, the columns are copied by a factor
    of 2 to match the 4 columns. confirm this is the desired functionality.

    FIXME: for shared keys, if to_ind goes out of bound, should to_meta's shared key be
    extended to incorporate that? or should it raise an error? Current version does not extend
    '''
    if isinstance(from_ind, int):
        from_ind = [from_ind]

    if isinstance(to_ind, int):
        to_ind = [to_ind]

    if not to_ind or not from_ind:
        # since to_ind and from_ind are lists for int inputs, 
        # no need to worry about an input of 0 for to_ind treated as an empty list
        # if to_ind or from_ind are empty, do nothing to to_meta
        return to_meta

    # NOTE: from_meta's fields should be arrays or 2d lists with fixed sizes
    # different 0th dimension's indices correspond to different columns in the MATLAB version
    for key, value in from_meta.items(): 
    # NOTE: loops through from_meta's keys. Thus, for to_meta's pure keys (keys that only exist
    # in to_meta but not from_meta), the values are identical
        if key in exclude: 
            continue
        
        if key in to_meta.keys():
            to_value = np.zeros((max(max(to_ind) + 1, len(to_meta[key])), value.shape[1]))
            to_value[:len(to_meta[key]), :] = np.array(to_meta[key])
            # the version below raises error later if to_ind goes out of to_meta[key]'s index
            # to_value = np.array(to_meta[key]) 
        else:
            to_value = np.zeros((max(to_ind)+1, value.shape[1]))
        to_value[to_ind, :] = np.tile(value[from_ind, :], [int(len(to_ind) / len(from_ind)), 1])
        to_meta[key] = to_value

    return to_meta

def conv_sub_1d(data, filt, ds):
    '''
    performs 1d convolution followed by downsampling in real space, which corresponds to multiplication followed by 
    periodization (when downsampling) or zeropadding (when upsampling). the returned signal is in real space.
    
    FIXME: only allow 'fourier_multires' option for simplicity if no significant speedup is shown. 
    check if this function is used in other functions with filt being given as list 

    inputs:
    -------
    - data: list or nparray dypte. data to be convolved in the frequency domain
    - filt: dict or list or nparray type. analysis filt in the frequency domain.
    if dict type, it has filt type and coef value. Based on the filt type (fourier_multires or fourier_truncated),
    the way the convolution is done differs
    - ds: downsampling factor exponent when represented in power of 2

    outputs:
    --------
    - y_ds: convolved signal in real space followed by downsampling 

    FIXME:try broadcasting instead of np.tile() followed by elementwise multiplication
    NOTE:seems like reading the MATLAB version, filt signal is always 1d array (either given as list or dict value)
    when filt is given as a dict, it will have multiple 1d filters. This function finds the filter whose size matches
    with that of the given data, and uses that filter for convolution, which is done by multiplying in fourier space in this case
    '''

    data = np.array(data)
    # for 1 signal, a singleton dim added
    if len(data.shape) == 1:
        data = data[np.newaxis, :] 
    n_data = data.shape[0]
    data_len = data.shape[1]
    # data is now shaped (n_data, data_len)

    # FIXME:NOTE: filt assumed to be either dict, list, or np.array. data assumed to be either np.array or list
    # FIXME: filt assumed to be rank 1
    if isinstance(filt, dict):
        # optimized filt, output of optimize_filter()
        if filt['type'] == 'fourier_multires':
            # NOTE: filt['coef'] is assumed to be a LIST of filters where each filter is rank 1
            # periodized multiresolution filt, output of PERIODIZE_FILTER
            # make coef into rank 2 array sized (1, filt_len)
            coef = filt['coef'][int(np.round(np.log2(filt['N'] / data_len)))]
            coef = np.array(coef)[np.newaxis, :] 
            # yf = data * np.tile(coef, (n_data, 1))
            yf = data * coef # broadcasting. Not tested yet.
        elif filt['type'] == 'fourier_truncated':
            # in this case, filt['coef'] is assumed to be an array 
            # truncated filt, output of TRUNCATE_FILTER
            start = filt['start']
            coef = filt['coef']
            n_coef = len(coef)
            coef = np.array(coef)
            if n_coef > data_len:
                # filt is larger than signal, lowpass filt & periodize the former
                # create lowpass filt
                start0 = start % filt['N']
                if (start0 + n_coef) <= filt['N']:
                    rng = np.arange(start0, n_coef - 1)
                else:
                    rng = np.concatenate([np.arange(start0, filt['N']), np.arange(n_coef + start0 - filt['N'])], axis=0)

                lowpass = np.zeros(n_coef)
                lowpass[rng < data_len / 2] = 1
                lowpass[rng == data_len / 2] = 1/2
                lowpass[rng == filt['N'] - data_len / 2] = 1/2
                lowpass[rng > filt['N'] - data_len / 2] = 1
                # filter and periodize
                coef = np.reshape(coef * lowpass, [data_len, int(n_coef / data_len)]).sum(axis=1)
                coef = coef[np.newaxis, :]

            j = int(np.round(np.log2(n_coef / data_len)))
            start = start % data_len
            if start + n_coef <= data_len:
                # filter support contained in one period, no wrap-around
                yf = data[:, start:n_coef+start-1] * np.tile(coef, (n_data, 1))
            else:
                # filter support wraps around, extract both parts
                yf = np.concatenate([data[:, start:], data[:, :n_coef + start - size(data,1)]], axis=1) * np.tile(coef, (n_data, 1))

    else:
        # type is either list or nparray
        filt = np.array(filt)
        # simple Fourier transform. filt_j below has length equal to data_len.
        # filt_j is a fraction taken from filt to match length with data_len.
        # if data_len is [10,11,12,13,14,15] and filt being range(100), 
        # filt_j would be [0, 1, 2, (3 + 98)/2, 99, 100].
        # REVIEW: figure out why the shifting is done before multiplying. 
        # Perhaps related to fftshift?
        # take only part of filter and shift it 
        filt_j = np.concatenate([filt[:int(data_len/2)],
            [filt[int(data_len / 2)] / 2 + filt[int(-data_len / 2)] / 2],
            filt[int(-data_len / 2 + 1):]], axis=0) # filt_j's length is identical to data_len
        filt_j = filt_j[np.newaxis, :] # shape: (1, data_len)
        yf = data * filt_j # will use broadcasting. 
        # FIXME: for 1 signal, a singleton dim added, resulting in yf being rank 2 array. return as in this shape?
    
    # calculate the downsampling factor with respect to yf
    dsj = int(ds + np.round(np.log2(yf.shape[1] / data_len)))
    if dsj > 0:
        # actually downsample, so periodize in Fourier
        yf_ds = np.reshape(yf, [n_data, int(2**dsj), int(np.round(yf.shape[1]/2**dsj))]).sum(axis=1)
    elif dsj < 0:
        # upsample, so zero-pad in Fourier
        # note that this only happens for fourier_truncated filters, since otherwise
        # filter sizes are always the same as the signal size
        # also, we have to do one-sided padding since otherwise we might break 
        # continuity of Fourier transform
        yf_ds = np.concatenate(yf, np.zeros(yf.shape[0], (2**(-dsj)-1)*yf.shape[1]), axis=1) # FIXME: not sure
    else: # if dsj == 0
        yf_ds = yf
    
    if isinstance(filt, dict) and filt['type'] == 'fourier_truncated' and filt['recenter']:
        # result has been shifted in frequency so that the zero fre-
        # quency is actually at -filt.start+1
        yf_ds = np.roll(yf_ds, filt['start']-1, axis=1)

    y_ds = np.fft.ifft(yf_ds, axis=1) / 2**(ds/2) # ifft default axis=-1
    # the 2**(ds/2) factor seems like normalization

    return y_ds

def pad_signal(data, pad_len, mode='symm', center=False):
    '''
    NOTE: assuming that data is given as either rank 1 or 2 array
    rank 1: (data_len,), rank 2: (n_data, data_len)

    NOTE: in the matlab version, having pad_len being a len 2 list means padding in both 2 directions.
    this means that this feature is for 2d signals, not for usage in multiple wavelets (Q can be a len 2 list ([8, 1])
    even in 1d signals case) and so I was worried if I have to write pad_signal() allowing pad_len to be len 2

    inputs:
    -------
    - data: rank 1 or 2 array or list with shape (data_len,) or (n_data, data_len)
    - pad_len: int type, length after zadding
    - mode: string type either symm, per, zero.
    - center: bool type indicating whether to bring the padded signal to the center

    outputs:
    --------
    - data: nparray type padded data shaped (n_data, pad_len)
    FIXME: if n_data is 1 in input argument, remove that dimension? for consistency, I think I should just make it (n_data, pad_len) even when n_data is 1
    '''
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[np.newaxis, :] # data is now rank 2 with shape (n_data, data_len)
    data_len = data.shape[1] # length of a single data.
    has_imag = np.linalg.norm(np.imag(data)) > 0 # bool type that checks if any of the elements has imaginary component

    if mode == 'symm':
        idx0 = np.concatenate([np.arange(data_len), np.arange(data_len, 0, -1) - 1], axis=0)
        conjugate0 = np.concatenate([np.zeros(data_len), np.ones(data_len)], axis=0)
    elif mode == 'per' or mode == 'zero':
        idx0 = np.arange(data_len)
        conjugate0 = np.zeros(data_len)
    else:
        raise ValueError('Invalid boundary conditions!')

    if mode != 'zero':
        idx = np.zeros(pad_len)
        conjugate = np.zeros(pad_len)
        idx[:data_len] = np.arange(data_len)
        conjugate[:data_len] = np.zeros(data_len)
        src = np.arange(data_len, data_len + np.floor((pad_len-data_len) / 2), dtype=int) % len(idx0)
        dst = np.arange(data_len, data_len + np.floor((pad_len - data_len) / 2), dtype=int)
        idx[dst] = idx0[src]
        conjugate[dst] = conjugate0[src]
        src = (len(idx0) - np.arange(1, np.ceil((pad_len - data_len) / 2) + 1, dtype=int)) % len(idx0)
        dst = np.arange(pad_len - 1, data_len + np.floor((pad_len - data_len) / 2 ) - 1, -1, dtype=int)
        idx[dst] = idx0[src]
        conjugate[dst] = conjugate0[src]
        # conjugate is shaped (pad_len,)
    else:
        idx = np.arange(data_len)
        conjugate = np.zeros(data_len)
        # conjugate is shaped (data_len,)

    idx = idx.astype(int)
    # idx, idx0, conjugate, conjugate0, src, dst are all rank 1 arrays
    data = data[:, idx] # data shape: (n_data, data_len or pad_len)
    conjugate = conjugate[np.newaxis, :] # conjugate shape: (1, data_len or pad_len)
    if has_imag:
        data = data - 2j * np.imag(data) * conjugate
    # data is shaped (n_data, data_len or pad_len)

    if mode == 'zero':
        data = np.concatenate([data, np.zeros((data.shape[0], pad_len - data_len))], axis=1)

    if center: # if center is nonzero (negative values are allowed, too)
        margin = int(np.floor((pad_len - data_len) / 2))
        data = np.roll(data, margin, axis=1)

    return data

def periodize_filter(filter_f):
    '''
    periodizes filter at multiple resolutions
    inputs:
    -------
    - filter_f: rank 1 nparray or list. filter in the frequency domain (indicated by _f)

    outputs:
    --------
    - coef: list whose elements are periodized filter at different resolutions. the output filters are in frequency domain

    NOTE: if filter_f has length N, the output coef is a list of nparrays with length being
    [N/2**0, N/2**1, N/2**2, ...] as far as N/2**n is an integer.
    for a filter at a specific resolution, if the length is l, this filter is approximately
    [the first l/2 values of filter_f, some value in the middle of filter_f, the last l/2 values of filter_f]
    REVIEW: check how this function is being used. Why for each resolution, take the lowest and highest coefficients?

    NOTE: filter_f is a rank 1 array. In the MATLAB version, a rank 2 array is allowed.
    this might correspond to 2 different filters for 1d signals when Q is a len 2 list. 
    However, if N is the shape of filter_f, the function does not only N(1)/2**j0 but also N(2)/2**j0
    which means that the meaning of N(2) is the length of the filter in that direction, NOT simply the number of filters
    Therefore we can use this function for filter_f being a rank 1 array'''
    filter_f = np.array(filter_f)
    N = len(filter_f)
    coef = []

    # j0 = 0
    n_filter = sum((N / 2.**(np.arange(1, np.log2(N)+1))) % 1 == 0)
    # n_filter is the number of times N can be divided with 2
    for j0 in range(n_filter):
        filter_fj = filter_f.copy()

        mask =  np.array([1.]*int(N/2.**(j0+1)) + [1./2.] + [0.]*int((1-2.**(-j0-1))*N-1))
        mask += np.array([0.]*int((1-2.**(-j0-1))*N) + [1./2.] + [1.]*int(N/2.**(j0+1)-1))

        filter_fj = filter_fj * mask
        filter_fj = np.reshape(filter_fj, [int(N/2**j0), 2**j0], order='F').sum(axis=1)
        coef.append(filter_fj)
        # j0 += 1

    return coef

def unpad_signal(data, res, orig_len, center=False):
    '''
    remove padding from the result of pad_signal()
    inputs:
    -------
    - data: list or np.array type data to be padded. either rank 1 (data_len,) or rank 2 (n_data, data_len)
    - res: int type indicating the resolution of the signal (exponent when expressed as power of 2)
    - orig_len: int type, length of the original, unpadded version. output is sized 
    - center: bool type indicating whether to center the output

    outputs:
    --------
    - data: nparray. either rank 1 (len,) or rank 2 (n_data, len) where len is
    orig_len*2*(-res)
    '''
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[np.newaxis, :] # data is now rank 2 with shape (n_data, data_len)

    offset = 0
    if center:
        offset = int((len(data) * 2**res - orig_len) / 2) # FIXME: test cases where the argument of int() is not exactly an integer. Or, should it always be an integer?
    
    offset_ds = int(np.floor(offset / 2**res))
    orig_len_ds = int(np.floor((orig_len-1) / 2**res)) # FIXME: what is it with the -1?
    
    data = data[:, offset_ds:offset_ds + orig_len_ds]

    return data

def wavelet_1d(data, filters, options={}):
    '''
    1d wavelet transform
    FIXME:REVIEW:review this function. Don't understand fully.
    inputs:
    -------
    - data: list or nparray shaped (data_len,) or (n_data, data_len)
    - filters: dict type object containing filter banks and their parameters
    - options: dict type object containing optional parameters
    '''
    options = fill_struct(options, oversampling=1)
    options = fill_struct(options, psi_mask=[True] * len(filters['psi']['filter'])) # FIXME: originally was true(1, numel(filters.psi.filter))
    options = fill_struct(options, data_resolution=0)

    data = np.array(data)
    if len(data.shape) == 1:
        data = data[np.newaxis, :] # data is now rank 2 with shape (n_data, data_len)
    data_len = data.shape[1]
    _, psi_bw, phi_bw = filter_freq(filters['meta']) # filter_freq returns xi_psi, bw_psi, bw_phi

    j0 = options['data_resolution']

    pad_len = filters['meta']['size_filter'] / 2**j0
    # pad_signal()'s arguments are data, pad_len, mode='symm', center=False
    data = pad_signal(data, pad_len, filters['meta']['boundary']) 

    xf = np.fft.fft(data, axis=1)
    
    ds = np.round(np.log2(2 * np.pi / phi_bw)) - j0 - options['oversampling']
    ds = max(ds, 0)
    
     
    x_phi = np.real(conv_sub_1d(xf, filters['phi']['filter'], ds)) # arguments should be in frequency domain. Check where filters['phi']['filter'] is set as the frequency domain
    x_phi = unpad_signal(x_phi, ds, data_len)
    # so far we padded the data (say the length became n1 -> n2) then calculated the fft of that to use as an input for conv_sub_1d()
    # the output has length n2 and so we run unpad_signal() to cut it down to length n1.
    meta_phi['j'] = -1 # REVIEW: why is j -1? what is j?
    meta_phi['bandwidth'] = phi_bw
    meta_phi['resolution'] = j0 + ds

    # x_psi = []
    x_psi = [None] * filters['psi']['filter'] # FIXME: replacing zeros(n,0) with this. This line might break
    meta_psi['j'] = -1 * np.ones(len(filters['psi']['filter'])) # REVIEW: why -1? what are these?
    meta_psi['bandwidth'] = -1 * np.ones(len(filters['psi']['filter']))
    meta_psi['resolution'] = -1 * np.ones(len(filters['psi']['filter']))
    for p1 in np.where(options['psi_mask'])[0]: # FIXME: options['psi_mask'] is a list of bool type elements
    # p1: indices where options['phi_mask'] is True
        ds = np.round(np.log2(2 * np.pi / psi_bw[p1] / 2)) - j0 - max(1, options['oversampling'])
        ds = max(ds, 0)

        x_psi_tmp = conv_sub_1d(xf, filters['psi']['filter'][p1], ds)
        x_psi[p1] = unpad_signal(x_psi_tmp, ds, data_len)
        meta_psi['j'][:, p1] = p1 # FIXME: might break: in matlab version this was = p1 - 1. I changed to = p1 instead.
        meta_psi['bandwidth'][:, p1] = psi_bw[p1]
        meta_psi['resolution'][:,p1] = j0 + ds

    if len(x_psi) != len(filters['psi']['filter']):
        raise ValueError("x_psi has different size from what it is expected. In MATLAB version it was initialized to cell array sized (1, filters['psi']['filter']). However, after appending all the elements to the empty list in python, the result is a list with length different from what is expected.")

    return x_phi, x_psi, meta_phi, meta_psi

def modulus_layer(W):
    U['signal'] = [np.abs(sig) for sig in W['signal']]
    U['meta'] = W['meta']
    return U

def scat(x, Wop): # NOTE:Wop should be a list of functions
    # Initialize signal and meta
    U[0]['signal'][0] = x
    U[0]['meta']['j'] = np.zeros(0,1);
    U[0]['meta']['q'] = np.zeros(0,1);
    U[0]['meta']['resolution'] = 0

    # Apply scattering, order per order
    for m in range(len(Wop)):
        if m < len(Wop) - 1:
            S[m], V = Wop[m](U[m])
            U[m] = modulus_layer(V)
        else:
            S[m] = Wop[m](U[m])

    return S, U

def wavelet_factory_1d(N, filt_opt=None, scat_opt=None):
    if filter_opt is None:
        filters = filter_bank(N)
    else:
        filters = filter_bank(N, filt_opt)
    
    if scat_opt is None:
        scat_opt = {} 
    scat_opt = fill_struct(scat_opt, M=2) # M is the scattering order
    
    Wop = [None] * scat_opt['M']
    for m in range(scat_opt['M'] + 1):
        filt_ind = min(len(filters) - 1, m);
        Wop[m] = lambda x: wavelet_layer_1d(x, filters[filt_ind], scat_opt)

    return Wop, filters




def wavelet_layer_1d(U, filters, scat_opt=None, wavelet=None):
    if scat_opt is None:
        scat_opt = {}

    if wavelet is None:
        wavelet = wavelet_1d
    
    scat_opt = fill_struct(scat_opt, path_margin=0)
    
    psi_xi, psi_bw, phi_bw = filter_freq(filters['meta'])
    
    if 'bandwidth' not in U['meta'].keys():
        U['meta']['bandwidth'] = 2 * np.pi
    if 'resolution' not in U['meta'].keys():
        U['meta']['resolution'] = 0
    
    U_phi['signal'] = {}
    U_phi['meta']['bandwidth'] = []
    U_phi['meta']['resolution'] = []
    U_phi['meta']['j'] = [None] * len(U['meta']['j']) # FIXME: replacing zeros(n,0) with this. This line might break
    
    U_psi.signal = {}
    U_psi['meta']['bandwidth'] = [];
    U_psi['meta']['resolution'] = [];
    U_psi['meta']['j'] = [None] * (len(U['meta']['j']) + 1) # FIXME: replacing zeros(n,0) with this. This line might break
    
    r = 0
    for p1 in range(len(U['signal'])):
        current_bw = U['meta']['bandwidth'][p1]*2**scat_opt['path_margin']
        psi_mask = current_bw > psi_xi
        
        scat_opt['x_resolution'] = U['meta']['resolution'][p1]
        scat_opt['psi_mask'] = psi_mask
        x_phi, x_psi, meta_phi, meta_psi = wavelet(U['signal'][p1], filters, scat_opt)
        
        U_phi['signal'][0, p1] = x_phi # FIXME: matlab version does U_phi.signal{1,p1}. This line might break
        U_phi['meta'] = map_meta(U['meta'], p1, U_phi['meta'], p1)
        U_phi['meta']['bandwidth'][0, p1] = meta_phi['bandwidth']
        U_phi['meta']['resolution'][0, p1] = meta_phi['resolution']
        
        ind = list(range(r, r + sum(psi_mask)))
        U_psi['signal'][0, ind] = x_psi[0, psi_mask]
        U_psi['meta'] = map_meta(U['meta'], p1, U_psi['meta'], ind, {'j'}) # FIXME: {'j'}? check how it'll work
        U_psi['meta']['bandwidth'][0, ind] = meta_psi['bandwidth'][0, psi_mask]
        U_psi['meta']['resolution'][0, ind] = meta_psi['resolution'][0, psi_mask]
        U_psi['meta']['j'][:, ind] = np.concatenate([np.dot(U['meta']['j'][:,p1], np.ones(1, len(ind))), meta_psi['j'][0,psi_mask]], axis=0) # FIXME: this line might break
            
        r += len(ind)
    return U_phi, U_psi

def filter_bank(data_len, options=None):
    parameter_fields = ['filter_type','Q','B','xi_psi','sigma_psi', 'phi_bw_multiplier','sigma_phi','J','P','spline_order', 'filter_format']
        
    if options is None:
        options = {}

    options = fill_struct(options, filter_type='morlet_1d')
    options = fill_struct(options, Q=[])
    options = fill_struct(options, B=[])
    options = fill_struct(options, xi_psi=[])
    options = fill_struct(options, sigma_psi=[])
    options = fill_struct(options, phi_bw_multiplier=[])
    options = fill_struct(options, sigma_phi=[])
    options = fill_struct(options, J=[])
    options = fill_struct(options, P=[])
    options = fill_struct(options, spline_order=[])
    options = fill_struct(options, precision='double')
    options = fill_struct(options, filter_format='fourier_truncated')
    
    if not isinstance(options.filter_type, list):
        options.filter_type = [options.filter_type]
    
    if not isinstance(options.filter_format, list):
        options.filter_format = [options.filter_format]
        
    bank_count = max([len(options[x]) for x in parameter_fields]) # number of required filter banks
        
    for k in range(bank_count):
        # extract the kth element from each parameter field
        options_k = options.copy()
        for l in range(len(parameter_fields)):
            if parameter_fields[l] not in options_k.keys():
                continue
            elif options_k[parameter_fields[l]] is None:
                continue
            value = options_k[parameter_fields[l]]
            # FIXME: this part below might break
            # in matlab, for example, it has Q = [8 1] (numeric) or cell arrays.
            # the isnumeric() part takes care of Q = [8 1] stuff, and 
            # iscell() part takes care of cell arrays.
            # in our case, they are both lists, and so we simply group the two cases into one.
            # FIXME: Q, B stuff should be lists but not np.array??
            if isinstance(value, (list, np.array)): # FIXME: allow np.array? 
                value_k = value[min(k, len(value))]
            else:
                print("options_k value type:{}".format(type(value)))
                value_k = None
            options_k[parameter_fields[l]] = value_k
        
        # calculate the kth filter bank
        if options_k['filter_type'] == 'morlet_1d' or options_k['filter_type'] == 'gabor_1d':
            filters[k] = morlet_filter_bank_1d(data_len, options_k)
        elif options_k['filter_type'] == 'spline_1d':
            filters[k] = spline_filter_bank_1d(data_len, options_k)
        elif options_k['filter_type'] == 'selesnick_1d':
            filters[k] = selesnick_filter_bank_1d(data_len, options_k)
        else:
            raise ValueError('Unknown wavelet type:{}'.format(options_k['filter_type']))

    return filters

def morlet_filter_bank_1d(data_len, options=None):
    if options is None:
        options = {}
    
    parameter_fields = ['filter_type','Q','B','J','P','xi_psi', 'sigma_psi', 'sigma_phi', 'boundary', 'phi_dirac']
    
    # If we are given a two-dimensional size, take first dimension
    data_len = data_len[-1]
    
    sigma0 = 2 / np.sqrt(3)
    
    # Fill in default parameters
    options = fill_struct(options, filter_type='morlet_1d')
    options = fill_struct(options, Q=1)
    options = fill_struct(options, B=options['Q'])
    options = fill_struct(options, xi_psi=1 / 2 * (2**(-1 / options['Q']) + 1) * np.pi)
    options = fill_struct(options, sigma_psi=1 / 2 * sigma0 / (1 - 2**(-1 / options['B'])))
    options = fill_struct(options, phi_bw_multiplier=1 + (options['Q'] == 1))
    options = fill_struct(options, sigma_phi=options['sigma_psi'] / options['phi_bw_multiplier'])
    options = fill_struct(options, J=T_to_J(data_len, options))
    options = fill_struct(options, P=np.round((2**(-1 / options['Q']) - 1 / 4 * sigma0 / options['sigma_phi']) / (1 - 2**(-1 / options['Q']))))
    options = fill_struct(options, precision='double')
    options = fill_struct(options, filter_format='fourier_truncated')
    options = fill_struct(options, boundary='symm')
    options = fill_struct(options, phi_dirac=0)
    
    if options['filter_type'] != 'morlet_1d' and options['filter_type'] != 'gabor_1d':
        raise ValueError('Filter type must be morlet_1d or gabor_1d')
    
    do_gabor = options['filter_type'] == 'gabor_1d'
    
    filters = {}
    
    # Copy filter parameters into filter structure. This is needed by the
    # scattering algorithm to calculate sampling, path space, etc.
    filters.meta = {}
    for l in range(len(parameter_fields)):
        filters['meta'][parameter_fields[l]] = options[parameter_fields[l]] # FIXME: check if things get connected...
    # FIXME: in the matlab version, you set the field and return the struct. Here, I simply update the dict.
    # should this do the job?

    # The normalization factor for the wavelets, calculated using the filters
    # at the finest resolution (N)
    psi_ampl = 1;
    
    if options['boundary'] == 'symm':
        N = 2 * data_len
    else:
        N = data_len
    
    N = int(2**np.ceil(np.log2(N)))
    
    filters['meta']['size_filter'] = N
    
    filters['psi']['filter'] = [None] * (options['J'] + options['P']) # FIXME: might break. empty cell arrays have been replaced with [None] * N arrays. Is this the right way?
    filters['phi'] = None # FIXME: Might break. Originally in MATLAB this is  = []
    
    psi_center, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    
    psi_sigma = sigma0 * np.pi / 2. / psi_bw
    phi_sigma = sigma0 * np.pi / 2. / phi_bw
    
    # Calculate normalization of filters so that sum of squares does not
    # exceed 2. This guarantees that the scattering transform is
    # contractive.
    S = np.zeros(N)
    
    # As it occupies a larger portion of the spectrum, it is more
    # important for the logarithmic portion of the filter bank to be
    # properly normalized, so we only sum their contributions.
    for j1 in range(options['J']):
        temp = gabor(N, psi_center[j1], psi_sigma[j1])
        if not do_gabor:
            temp = morletify(temp,psi_sigma[j1])
        S = S + np.abs(temp)**2;
    
    psi_ampl = np.sqrt(2 / max(S))
    
    # Apply the normalization factor to the filters.
    for j1 in range(len(filters['psi']['filter'])):
        temp = gabor(N, psi_center[j1], psi_sigma[j1])
        if not do_gabor:
            temp = morletify(temp,psi_sigma[j1])
        filters['psi']['filter'][j1] = optimize_filter(psi_ampl * temp, 0, options)
        filters['psi']['meta']['k'][j1, 0] = j1
    # Calculate the associated low-pass filter
    if not options['phi_dirac']:
        filters['phi']['filter'] = gabor(N, 0, phi_sigma)
    else:
        filters['phi']['filter'] = np.ones(N,1)
    
    filters['phi']['filter'] = optimize_filter(filters['phi']['filter'], 1, options)
    filters['phi']['meta']['k'][0, 0] = options['J'] + options['P']

    return filters

def gabor(N, xi, sigma):
    extent = 1 # extent of periodization - the higher, the better
    sigma = 1 / sigma
    f = np.zeros(N)
    
    # calculate the 2*pi-periodization of the filter over 0 to 2*pi*(N-1)/N
    for k in range(-extent, 2 + extent):
        f += np.exp(-((np.arange(N) - k * N) / N * 2 * np.pi - xi)**2. / (2 * sigma**2))
    return f

def morletify(f, sigma):
    f0 = f[1]
    f = f - f0 * gabor(len(f), 0, sigma)
    return f

def dyadic_freq_1d(filter_options):
    pass

