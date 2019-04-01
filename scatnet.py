'''module for performing invariant scattering transformation on multichannel time series'''
import os
import numpy as np
import pdb
import copy
'''
FIXME: consider name change and check comments for the following functions by comparing with MATLAB version
_morlet_freq_1d, _morlet_filter_bank_1d, _conv_sub_1d

FIXME: read script and check if any of the attribute objects get linked to something during function call

FIXME: for functions that allow signal to be rank 1 array for number of data being 1, only allow rank 2 inputs
(even for 1 signal, it should be shaped (1, data_len)). This is important as we want to add the channel dimension as well,
although calculating and combining different channels might be possible to be done at a higher level (such as in function scat(), etc)

FIXME: namechange: remove "1d" in all variables, function names
'''


class ScatNet(object):
    def __init__(self, data_len, avg_len, n_filter_octave=[8, 1], filter_format='fourier_multires', mode='symm'):
        if isinstance(n_filter_octave, int):
            n_filter_octave = list(n_filter_octave)

        self._data_len = data_len
        self._avg_len = avg_len
        self._n_filter_octave = np.array(n_filter_octave)
        self._n_layers = len(n_filter_octave)

        self._filter_format = filter_format
        self._bw_recip_octave = np.array(n_filter_octave)
        self._mode = mode

        self._phi_bw_multiplier = self._get_phi_bw_multiplier(n_filter_octave=self._n_filter_octave)
        self._sigma_psi = self._get_sigma_psi(bw_recip_octave=self._bw_recip_octave)
        self._sigma_phi = self._get_sigma_phi(sigma_psi=self._sigma_psi, phi_bw_multiplier=self._phi_bw_multiplier)

        self._n_filter_log = self._get_n_filter_log(avg_len=self._avg_len, n_filter_octave=self._n_filter_octave)
        self._n_filter_lin = self._get_n_filter_lin(sigma_phi=self._sigma_phi, n_filter_octave=self._n_filter_octave)
        self._xi_psi = self._get_xi_psi(n_filter_octave=self._n_filter_octave)

        # everything so far is correct for data length 64, T=32
        self._filters = self._get_filters()

        # REVIEW: path_margin, oversampling, truncated_threshold were removed from attributes

    # FIXME: add a function that prints out the following attributes:
    # data_len, avg_len, n_layers, n_filter_octave, n_filter_log, n_filter_lin, sigma_psi, sigma_phi, xi_psi, bw_recip_octave, filter_format, mode

    @property
    def data_len(self):
        '''getter for the allowed length of data'''
        return self._data_len

    @property
    def avg_len(self):
        '''getter for scaling function window length'''
        return self._avg_len

    @property
    def n_filter_octave(self):
        '''getter for number of filters per octave'''
        return self._n_filter_octave

    @property
    def n_layers(self):
        '''getter for number of layers'''
        return self._n_layers

    def _get_n_filter_lin(self, sigma_phi, n_filter_octave):
        '''
        calculates n_filter_lin, the number of filters that are linearly spaced
        '''
        sigma0 = 2 / np.sqrt(3)
        n_filter_lin = np.round((2**(-1 / n_filter_octave) - 1 / 4 * sigma0 / sigma_phi) / (1 - 2**(-1 / n_filter_octave))).astype(int)
        assert(n_filter_lin >= 0).all(), 'Invalid number of linearly spaced filters. n_filter_lin should be nonnegative'

        return n_filter_lin

    def _get_n_filter_log(self, avg_len, n_filter_octave):
        '''
        calculates maximal wavelet scale n_filter_log, the number of filters that are logarithmically spaced
        '''
        phi_bw_multiplier = (1 + (n_filter_octave==1)).astype('int')
        n_filter_log = 1 + np.round(np.log2(avg_len / (  4 * n_filter_octave / phi_bw_multiplier)) * n_filter_octave).astype(int)
        assert(n_filter_log > 0).all(), 'Invalid number of logarithmically spaced filters. n_filter_log should be positive'

        return n_filter_log

    def _get_sigma_psi(self, bw_recip_octave):
        '''
        calculates sigma_psi, the standard deviation of the mother wavelet in space
        '''
        sigma0 = 2 / np.sqrt(3)
        sigma_psi = 1 / 2 * sigma0 / (1 - 2**(-1 / bw_recip_octave)) # FIXME: in the same reason as above, this might break, so changed to nparray

        return sigma_psi

    def _get_phi_bw_multiplier(self, n_filter_octave):
        '''
        calculates phi_bw_multiplier, the multiplier that relates sigma_phi and sigma_psi
        '''
        phi_bw_multiplier = (1 + (n_filter_octave == 1)).astype(int)

        return phi_bw_multiplier

    def _get_sigma_phi(self, sigma_psi, phi_bw_multiplier):
        '''
        calculates sigma_phi from sigma_psi and the multiplier
        '''
        sigma_phi = sigma_psi / phi_bw_multiplier

        return sigma_phi

    def _get_xi_psi(self, n_filter_octave):
        '''
        calculates xi_psi from n_filter_octave
        '''
        xi_psi = 1 / 2 * (2**(-1 / n_filter_octave) + 1) * np.pi

        return xi_psi

    def _get_filters(self):
        '''
        generates filters from the attributes
        '''
        n_layers = self._n_layers
        filters = []
        for m in range(n_layers):
            filter_m = self._morlet_filter_bank_1d(m)
            filters.append(filter_m)

        return filters

    def _morlet_freq_1d(self, filter_options):
        '''
        given filter options, returns center frequencies and bandwidths of morlet filter banks
        
        inputs:
        ------- 
        - filter_options: type dict with the following keys:
        xi_psi, sigma_psi, sigma_phi, n_filter_log, n_filter_octave, n_filter_lin: all scalars
        These are values of the corresponding instance attributes at a specific layer

        outputs:
        -------- 
        - xi_psi: nparray sized (n_filter_log + n_filter_lin,)
        - bw_psi: nparray sized (n_filter_log + n_filter_lin + 1,)
        - bw_phi: float
        
        increasing index corresponds to filters with decreasing center frequency
        filters with high freq are logarithmically spaced, low freq interval is covered linearly
        NOTE: xi_psi can in theory have negative center frequencies in this function
        
        FIXME: consider defining local variables for the key-value pairs in filter_options during calculations
        '''
        sigma0 = 2 / np.sqrt(3)
        
        n_filter_log = filter_options['n_filter_log']
        n_filter_octave = filter_options['n_filter_octave']
        n_filter_lin = filter_options['n_filter_lin']
        
        # logarithmically spaced filters
        xi_psi = filter_options['xi_psi'] * 2**(np.arange(0,-n_filter_log,-1) / n_filter_octave)
        sigma_psi = filter_options['sigma_psi'] * 2**(np.arange(n_filter_log) / n_filter_octave)
        
        # linearly spaced filters
        if n_filter_lin != 0:
            step = np.pi * 2**(-n_filter_log / n_filter_octave) * (1 - 1/4 * sigma0 / filter_options['sigma_phi'] \
                * 2**( 1 / n_filter_octave ) ) / n_filter_lin

            xi_psi_lin = filter_options['xi_psi'] * 2**((-n_filter_log+1) / n_filter_octave) \
                - step * np.arange(1, n_filter_lin + 1)
            xi_psi = np.concatenate([xi_psi, xi_psi_lin], axis=0) 

        sigma_psi_lin = np.full((1 + n_filter_lin,), fill_value=filter_options['sigma_psi'] \
            * 2**((n_filter_log - 1) / n_filter_octave))
        sigma_psi = np.concatenate([sigma_psi, sigma_psi_lin], axis=0)
        # calculate scaling function bandwidth
        sigma_phi = filter_options['sigma_phi'] * 2**((n_filter_log-1) / n_filter_octave)
        # calculate frequential bandwidths from spatial bandwidths
        bw_psi = np.pi / 2 * sigma0 / sigma_psi
        bw_phi = np.pi / 2 * sigma0 / sigma_phi

        return xi_psi, bw_psi, bw_phi

    def _optimize_filter(self, filter):
        '''
        generate filters based on filter format

        inputs:
        -------
        - filter: rank 1 nparray type filter in the frequency domain

        outputs:
        --------
        - filter: np.array type (if fourier) or list type (if fourier_multires) or dict type (if fourier_truncated).
        If list type, the elements are nparrays with different lengths, corresponding to different resolutions
        If dict type, the key 'coef' contains the filter which is a rank 1 np.array

        FIXME: consider deprecating
        '''
        filter_format = self._filter_format

        if filter_format == 'fourier':
            filter = np.array(filter) # converting to nparray since multires case return type has to be np.array
        elif filter_format == 'fourier_multires':
            filter = self._periodize_filter(filter)
        elif filter_format == 'fourier_truncated':
            filter = self._truncate_filter(filter)
        else:
            raise ValueError('Unknown filter format {}'.format(filter_format))
        return filter

    def _conv_sub_1d(self, data, filter, ds):
        '''
        performs 1d convolution followed by downsampling in real space, given data and filters in frequency domain.
        This corresponds to multiplication followed by 
        periodization (when downsampling) or zeropadding (when upsampling). the returned signal is in real space.
        
        inputs:
        -------
        - data: np.array with shape (n_data, data_len). data to be convolved given in the frequency domain
        - filter: dict or nparray type given in the frequency domain.
        if filter_format is fourier_multires:
        filter is dict type with the following keys: 'coef', 'filter_len'  
        filter['coef'] is assumed to be a rank 1 list of filters where each filter is rank 1
        From the data length and filter['filter_len'], the array with the same size with that of the data is found and convolved.
        This dict type object filter is an output of _periodize_filter().

        if filter_format is fourier_truncated:
        filter is dict type with the following keys: 'coef', 'filter_len', 'start'
        filter['coef'] is assumed to be a rank 1 np.array
        
        - ds: downsampling factor exponent when represented in power of 2

        outputs:
        --------
        - y_ds: convolved signal in real space followed by downsampling having shape (n_data, data_output_len)
        
        FIXME: understand the idea of wrapping around, and other questions in the comments
        '''

        n_data = data.shape[0]
        data_len = data.shape[1]

        filter_format = self._filter_format
        if isinstance(filter, dict):
            if filter_format == 'fourier_multires':
                # _periodize_filter() generates a set of filters whose lengths are filter_len/2^0, filter_len/2^1, filter_len/2^2, ...
                # these filters are grouped into a rank 1 list. therefore, given the original size of the filter filter_len,
                # and the length of the data, the length of the filter can be determined which can be found from the list
                coef = filter['coef'][int(np.round(np.log2(filter['filter_len'] / data_len)))]
                # make coef into rank 2 array sized (1, filter_len) for broadcasting
                coef = coef[np.newaxis, :] 
                yf = data * coef 
            elif filter_format == 'fourier_truncated':
                # in this case, filter['coef'] is an np.array
                start = filter['start']
                coef = filter['coef']
                n_coef = len(coef)
                if n_coef > data_len:
                    # if filter is larger than the given data, create a lowpass filter and periodize it
                    # FIXME: what is the basis of this idea?
                    start0 = start % filter['filter_len']
                    if (start0 + n_coef) <= filter['filter_len']:
                        rng = np.arange(start0, n_coef - 1).astype(int)
                    else:
                        rng = np.concatenate([np.arange(start0, filter['filter_len']), np.arange(n_coef + start0 - filter['filter_len'])], axis=0).astype(int)

                    lowpass = np.zeros(n_coef)
                    lowpass[rng < int(data_len / 2)] = 1
                    lowpass[rng == int(data_len / 2)] = 1/2
                    lowpass[rng == int(filter['filter_len'] - data_len / 2)] = 1/2
                    lowpass[rng > int(filter['filter_len'] - data_len / 2)] = 1

                    # filter and periodize
                    coef = np.reshape(coef * lowpass, [int(n_coef / data_len), data_len]).sum(axis=0)
                # coef is rank 1
                n_coef = len(coef)
                coef = coef[np.newaxis, :]
                j = int(np.round(np.log2(n_coef / data_len)))
                start = start % data_len

                if start + n_coef <= data_len:
                    # filter support contained in one period, no wrap-around
                    yf = data[:, start:n_coef+start] * coef
                else:
                    # filter support wraps around, extract both parts
                    yf = np.concatenate([data[:, start:], data[:, :n_coef + start - data_len]], axis=1) * coef

        else:
            # filter is np.array type. perform fourier transform.
            # filter_j is a fraction taken from filter to match length with data_len.
            # if data_len is [10,11,12,13,14,15] and filter being range(100), 
            # filter_j would be [0, 1, 2, (3 + 98)/2, 99, 100].
            # REVIEW: figure out why the shifting is done before multiplying. 
            # Perhaps related to fftshift?
            # take only part of filter and shift it 
            filter_j = np.concatenate([filter[:int(data_len/2)],
                [filter[int(data_len / 2)] / 2 + filter[int(-data_len / 2)] / 2],
                filter[int(-data_len / 2 + 1):]], axis=0)
            filter_j = filter_j[np.newaxis, :] # shaped (1, data_len)
            yf = data * filter_j
        
        # calculate the downsampling factor with respect to yf
        dsj = ds + np.round(np.log2(yf.shape[1] / data_len))
        assert(float(dsj).is_integer()), "dsj should be an integer"

        if dsj > 0:
            # downsample (periodize in Fourier)
            # REVIEW: don't understand why reshape and sum things. Why is this downsampling in real space? (review 6.003 notes)
            # I tested and see that this is correct. try running the following in a notebook:
            # a = np.sin(np.linspace(0,100,10000)) + np.sin(np.linspace(0,300,10000)); af = np.fft.fft(a[np.newaxis, :]); af2 = np.reshape(af, [4,2500]).sum(axis=0); a2 = np.fft.ifft(af2)
            # fig1,ax1 = plt.subplots(); fig2,ax2 = plt.subplots(); ax1.plot(a); ax2.plot(np.real(a2))
            yf_ds = np.reshape(yf, [n_data, int(2**dsj), int(np.round(yf.shape[1]/2**dsj))]).sum(axis=1)
        elif dsj < 0:
            # upsample (zero-pad in Fourier)
            # note that this only happens for fourier_truncated filters, since otherwise
            # filter sizes are always the same as the signal size
            # also, we have to do one-sided padding since otherwise we might break 
            # continuity of Fourier transform
            yf_ds = np.concatenate(yf, np.zeros(yf.shape[0], (2**(-dsj)-1)*yf.shape[1]), axis=1)
        else:
            yf_ds = yf
        if isinstance(filter, dict):
            if filter_format == 'fourier_truncated':
                # result has been shifted in frequency so that the zero frequency is actually at -filter.start+1

                # always recenter if fourier_truncated
                yf_ds = np.roll(yf_ds, filter['start'], axis=1)

        y_ds = np.fft.ifft(yf_ds, axis=1) / 2**(ds/2)
        # the 2**(ds/2) factor seems like normalization

        return y_ds

    def _pad_signal(self, data, pad_len, mode='symm', center=False):
        '''
        inputs:
        -------
        - data: rank 2 np.array with shape (n_data, data_len)
        - pad_len: int type, length after padding
        - mode: string type either symm, per, zero.
        - center: bool type indicating whether to bring the padded signal to the center

        outputs:
        --------
        - data: nparray type padded data shaped (n_data, pad_len)
        '''
        data_len = data.shape[1]
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

    def _periodize_filter(self, filter):
        '''
        generates filters at multiple resolutions

        inputs:
        -------
        - filter: rank 1 np.array given in the frequency domain

        outputs:
        --------
        - filter: dict type object with the following fields:
        filter_len: int type, length of the given filter
        coef: list whose elements are periodized filter at different resolutions. the output filters are in frequency domain

        REVIEW: the basis of this function's idea?
        '''

        filter_len = len(filter)
        
        filters = {'filter_len':filter_len}

        coef = []
        n_filter = sum((filter_len / 2.**(np.arange(1, np.log2(filter_len)+1))) % 1 == 0)
        # n_filter is the number of times filter_len can be divided with 2
        for j0 in range(n_filter):
            filter_j = filter

            mask =  np.array([1.]*int(filter_len/2.**(j0+1)) + [1./2.] + [0.]*int((1-2.**(-j0-1))*filter_len-1))
            mask += np.array([0.]*int((1-2.**(-j0-1))*filter_len) + [1./2.] + [1.]*int(filter_len/2.**(j0+1)-1))

            filter_j = filter_j * mask
            filter_j = np.reshape(filter_j, [int(filter_len/2**j0), 2**j0], order='F').sum(axis=1)
            coef.append(filter_j)

        filters['coef'] = coef

        return filters

    def _unpad_signal(self, data, res, orig_len, center=False):
        '''
        remove padding from the result of _pad_signal()

        inputs:
        -------
        - data: rank 2 np.array data to be padded, shaped (n_data, data_len)
        - res: int type indicating the resolution of the signal (exponent when expressed as power of 2)
        - orig_len: int type, length of the original, unpadded version (but at different resolution). 
        - center: bool type indicating whether to center the output

        outputs:
        --------
        - data: rank 2 nparray shaped (n_data, len) where len is orig_len*2*(-res)
        '''
        data_len = data.shape[1] # length of a single data.

        offset = 0
        if center:
            offset = int((data_len * 2**res - orig_len) / 2)
        offset_ds = int(np.floor(offset / 2**res))
        orig_len_ds = 1 + int(np.floor((orig_len-1) / 2**res)) 
        # although this is an index, the value is identical to that of the matlab version since it is used for accessing values through [...:orig_len_ds]
        # but in python indexing the last index does not get included and so for this value we do not subtract 1 to get 0 based index.
        data = data[:, offset_ds:offset_ds + orig_len_ds]

        return data

    def _wavelet_1d(self, data, filters, psi_mask=None, x_res=0, oversampling=1):
        '''
        1d wavelet transform of the given data. The data is convolved with the scaling function and a subset of filter banks so that only
        frequency decreasing paths are considered. This corresponds to expanding a given node to branches in the graphical representation.

        used in _wavelet_layer_1d()
        
        FIXME:REVIEW:review this function. Don't understand fully.
        inputs:
        -------
        - data: list or nparray shaped (data_len,) or (n_data, data_len)
        - filters: dict type object containing filter banks and their parameters. this has the following keys:
        'psi' - dict type object containing the key 'filter'
        'phi' - dict type object containing the key 'filter'
        'meta' - dict type object containing keys 'size_filter' and 'boundary'

        - options: dict type object containing optional parameters. this has the following keys:
        'resolution', 'oversampling', 'psi_mask'. the key 'psi_mask' determines which filter banks will be used so that only 
        frequency decreasing paths are computed. This is given as an input argument.

        outputs:
        --------
        - x_phi: nparray shaped (n_data, data_len). data convolved with scaling function, represented in real space
        This n_data will later be used for the number of channels. This is NOT the number of nodes at a certain depth. This function is called within a for loop
        in the function _wavelet_layer_1d()
        - x_psi: rank 1 list of nparrays where each nparray is shaped (n_data, data_len). This is the data convolved with filters (psi) at multiple resolutions.
        For filters with whose corresponding value is False in options['psi_mask'], the convolution is skipped and gives a None value element in the list.
        - meta_phi, meta_psi: both dict type objects containing the convolution meta data for phi, psi, respectively. keys are 'j', 'bandwidth', 'resolution'
        For meta_phi, the values of the keys are scalars whereas for meta_psi, the values are all nparrays
        FIXME: meta_psi values: change to lists instead of nparrays?
        '''
        # options = copy.deepcopy(options) # FIXME: try avoiding this if possible
        filters = copy.deepcopy(filters)
        # options = fill_struct(options, oversampling=1)
        # options = fill_struct(options, psi_mask=[True] * len(filters['psi']['filter'])) # FIXME: in matlab, this is true(1, numel(filters.psi.filter))
        if psi_mask is None:
            psi_mask = [True] * len(filters['psi']['filter'])

        # filters['psi']['filter']
        # options = fill_struct(options, x_resolution=0)

        # data = np.array(data)
        # if len(data.shape) == 1:
        #     data = data[np.newaxis, :] # data is now rank 2 with shape (n_data, data_len)
        data_len = data.shape[1]

        # print("filters['meta']:{}".format(filters['meta']))
        # print("filters['psi']['filter']:{}".format(filters['psi']['filter']))
        _, psi_bw, phi_bw = self._morlet_freq_1d(filters['meta']) # _morlet_freq_1d returns psi_xi, psi_bw, phi_bw
        # print("psi_bw:{}".format(psi_bw))
        # print("phi_bw:{}".format(phi_bw))
        # j0 = options['x_resolution']
        j0 = x_res

        pad_len = int(filters['meta']['size_filter'] / 2**j0) # FIXME: size_filter is the only field in meta. way to get rid of this or perhaps remove meta?
        # _pad_signal()'s arguments are data, pad_len, mode='symm', center=False
        mode = self._mode
        data = self._pad_signal(data, pad_len, mode) 

        xf = np.fft.fft(data, axis=1)
        
        ds = int(np.round(np.log2(2 * np.pi / phi_bw)) - j0 - oversampling) # REVIEW: don't understand
        ds = max(ds, 0)
        
         
        x_phi = np.real(self._conv_sub_1d(xf, filters['phi']['filter'], ds)) # arguments should be in frequency domain. Check where filters['phi']['filter'] is set as the frequency domain
        x_phi = self._unpad_signal(x_phi, ds, data_len)

        # print("psi_bw:{}".format(psi_bw))
        # print("phi_bw:{}".format(phi_bw))
        # print("x_phi:{}".format(x_phi))
        # FIXME: in matlab, there's reshaping done: x_phi = reshape(x_phi, [size(x_phi,1) 1 size(x_phi,2)]);
        # IMPORTANT: check if this reshaping needs to be done

        # so far we padded the data (say the length became n1 -> n2) then calculated the fft of that to use as an input for _conv_sub_1d()
        # the output is in realspace, has length n2 and so we run _unpad_signal() to cut it down to length n1.
        meta_phi = {'j':-1, 'bandwidth':phi_bw, 'resolution':j0 + ds} # REVIEW: -1 for j, is this to denote that the value is empty or is this a valid value?
        # the j field does not get passed down eventually. In _wavelet_layer_1d, only fields bandwidth and resolution are passed on

        # x_psi = []
        x_psi = [None] * len(filters['psi']['filter']) # FIXME: replacing x_psi = cell(1, numel(filters.psi.filter)) with this. This line might break
        # meta_psi = {'j':-1 * np.ones((1, len(filters['psi']['filter'])))} # REVIEW:-1 is to denote that the value is empty (same for bandwidth and resolution) since you can't have -1 for these keys
        meta_psi = {'j':[-1] * len(filters['psi']['filter']) } # REVIEW:-1 is to denote that the value is empty (same for bandwidth and resolution) since you can't have -1 for these keys
        # meta_psi['bandwidth'] = -1 * np.ones((1, len(filters['psi']['filter'])))
        # meta_psi['resolution'] = -1 * np.ones((1, len(filters['psi']['filter'])))
        meta_psi['bandwidth'] = [-1] * len(filters['psi']['filter'])
        meta_psi['resolution'] = [-1] * len(filters['psi']['filter'])
        # for p1 in np.where(options['psi_mask'])[0]: # FIXME: options['psi_mask'] is a list of bool type elements
        for p1 in np.where(psi_mask)[0]: # FIXME: options['psi_mask'] is a list of bool type elements
        # p1: indices where options['psi_mask'] is True
            ds = np.round(np.log2(2 * np.pi / psi_bw[p1] / 2)) - j0 - max(1, oversampling) # FIXME: might break. what is 1 in max(1, options...)??
            ds = int(max(ds, 0))

            x_psi_tmp = self._conv_sub_1d(xf, filters['psi']['filter'][p1], ds)
            x_psi[p1] = self._unpad_signal(x_psi_tmp, ds, data_len)
            # print(meta_psi)
            # print(p1)
            meta_psi['j'][p1] = p1 # FIXME: might break: in matlab version this was = p1 - 1. I changed to = p1 instead.
            meta_psi['bandwidth'][p1] = psi_bw[p1] # FIXME: might break, in matlab version the LHS is meta_psi.bandwidth(:, p1). I don't know why
            meta_psi['resolution'][p1] = j0 + ds

        if len(x_psi) != len(filters['psi']['filter']): # FIXME: to be deprecated
            raise ValueError("x_psi has different size from what it is expected. In MATLAB version it was initialized to cell array sized (1, filters['psi']['filter']). However, after appending all the elements to the empty list in python, the result is a list with length different from what is expected.")

        return x_phi, x_psi, meta_phi, meta_psi

    def _modulus_layer(self, W):
        '''
        for data convolved with phi and psi at multiple resolutions at a given single layer, computes the modulus

        used in transform()

        inputs:
        -------
        - W: dict type object with key signal, meta where signal is a list of nparrays. For each array, the modulus is computed

        outputs:
        --------
        - U: dict type object with key signal, meta. for signal, the arrays are the modulus results of W['signal']
        '''
        U = {}
        U['signal'] = [np.abs(sig) for sig in W['signal']]
        U['meta'] = W['meta']
        return U

    def transform(self, data):
        '''
        compute the scattering transform

        used in main()

        inputs:
        -------
        - data: rank 1 list or nparray shaped (data_len,)
        - Wop: list of functions corresponding to convolution with phi, convolution with psi at multiple resolutions. This operator does not take the modulus

        outputs:
        --------
        - S: list type of... FIXME: write this part after reading other functions
        - U: list type where each element is a dict type with keys 'signal' and 'meta' and the index is the layer. this operator includes taking the modulus
        both values of 'signal' and 'meta' are lists where the index within denotes the scattering transform (convolution with phi, convolution + modulus with psi at multiple resolutions)
        FIXME: consider if Wop as an argument is necessary. Retain a copy of this version, but check with other demos and see if other types of Wop are used. If not, just remove that input argument and use the output of wavelet_factory_1d(). 
        FIXME: I think we already have an answer. The function that takes _wavelet_1d() as the default function handle is _wavelet_layer_1d(). This is called in wavelet_factory_1d(), which uses _wavelet_1d() as default when calling _wavelet_layer_1d()! Therefore, since _wavelet_1d() is default, simply just change things accordingly. For this transform() function, you should fix the 2nd argument of Wop here (corresponds to return_U)
        '''
        if len(data.shape) == 1:
            data = data[np.newaxis, :] 
        data_len = data.shape[1]

        # if filter_options is None:
            # filters = self._filter_bank(data_len)
        # else:
        #     filters = self._filter_bank(data_len, filter_options)
        
        # scat_options = fill_struct(scat_options, M=2) # M is the maximum scattering depth. FIXME: remove this
        n_layers = self._n_layers
        filters = self._filters

        S = []
        U_0 = {'signal':[data], 'meta':{'j':[], 'resolution':[0]}} # FIXME: removed q key
        U = [U_0]
        
        # Apply scattering, order per order
        for m in range(n_layers + 1):
            filt_ind = min(m, len(filters) - 1)
            # pdb.set_trace()
            if m < n_layers: # if this is not the last layer,
                S_m, V = self._wavelet_layer_1d(U=U[m], filters=filters[filt_ind])
                S.append(S_m)
                
                # S_m, V = Wop[m](copy.deepcopy(U[m]), True) # 2nd argument is for return_U. FIXME:change to more readable code
                # FIXME: remove copy.deepcopy() later
                # FIXME: test if Wop doesn't change the arguments.
                
                U.append(self._modulus_layer(V)) # NOTE: replaced U[m+1] = _modulus_layer(V) with this line. I think this will not break since both current and previous implementation adds at least something to the list S and U for len(Wop) times
            # else: # if this is the last layer, only compute S since V won't be used
            #     S_m = Wop[m](copy.deepcopy(U[m]), False) # 2nd argument is for return_U. FIXME:change to more readable code
            #     S.append(S_m)
            else:
                S_m, _ = self._wavelet_layer_1d(U=U[m], filters=filters[filt_ind])
                S.append(S_m)



        # for m in range(scat_options['M'] + 1): # I think this will not break since this for loop runs M+1 times and it always adds at least something to the list, which is being done with the append() function here instead of the Wop[m] = ... original implementation
        #     filter_ind = min(len(filters) - 1, m)

        #     Wop_m = lambda U, return_U: _wavelet_layer_1d(U=copy.deepcopy(U), filters=copy.deepcopy(filters[filter_ind]), scat_options=copy.deepcopy(scat_options), return_U=return_U)
        #     Wop.append(Wop_m)

        # return Wop, filters
            # print(len(S))    
            # print(len(U))    




        return S, U


    # def _wavelet_layer_1d(U, filters, return_U=True):
    def _wavelet_layer_1d(self, U, filters, path_margin=0):
        '''
        computes the 1d wavelet transform from the modulus. 
        _wavelet_1d() returns a list of signals (convolution at multiple resolutions) where this function uses the outputs of _wavelet_1d() and organizes them into proper data structures
        
        used in wavelet_factory_1d()

        FIXME: comeback to this function. don't fully understand it        
       
            
        inputs:
        -------
        - U: dict type object with input layer to be transformed. has the following keys:
        'meta': dict type object, has keys ('bandwidth', 'resolution', 'j')  whose values are (rank1, rank1, rank2) lists, respectively.
        'signal':rank 1 list type corresponding to the signals to be convolved. different signals correspond to different nodes which in this function will be convolved with phi and psi's.
        - filters: dict type object with the key 'meta'.
        - scat_options:
        - wavelet: function indicating wavelet transform. default is _wavelet_1d()

        outputs:
        --------
        - U_phi: dict type with following fields:
        'meta': dict type object, has keys ('bandwidth', 'resolution', 'j')  whose values are (rank1, rank1, FIXME:DON'T KNOW:rank1?) lists, respectively.
        'signal':rank 1 list type corresponding to the signals to be convolved # FIXME: not sure if this is rank 1...check where this function is used and what the input is.
            
        
            
            
        '''
        # scat_options = copy.deepcopy(scat_options)
        filters = copy.deepcopy(filters)
        U = copy.deepcopy(U) # FIXME: remove all these .copy() stuff later by only storing the necessary fields of U into a variable.
        # scat_options = fill_struct(scat_options, path_margin=0)
        psi_xi, psi_bw, phi_bw = self._morlet_freq_1d(filters['meta'])
        if 'bandwidth' not in U['meta'].keys():
            U['meta']['bandwidth'] = [2 * np.pi] # FIXME: confirm if this is right
            # NOTE: the reason we initialize it as a list is not becauase S['meta']['bandwidth'] will be set to a rank 2 list.
            # when running the invariant scattering transform, the resulting ['meta']['bandwidth'] will always be a rank 1 list.
            # the reason is because in matlab it's done as U.meta.bandwidth = 2*pi and in this case they treat it as a length 1 array
            # (all scalars in matlab are arrays with length 1 and can be accessed with c(1) if c is a scalar)
            # the same argument holds for the following ['meta']['resolution']
            # Note that ['meta']['j'] will be a rank 2 list since it shows the full path starting from the root node in the graphical representation.
        if 'resolution' not in U['meta'].keys():
            U['meta']['resolution'] = [0] # FIXME: confirm if this is right
        # pdb.set_trace()
        
        U_phi = {'signal':[], 'meta':{}}
        U_phi['meta']['bandwidth'] = []
        U_phi['meta']['resolution'] = []
        U_phi['meta']['j'] = [] # FIXME: for this one, the elements can be scalars or length 2 lists.

        
        U_psi = {'signal':[], 'meta':{}}
        U_psi['meta']['bandwidth'] = []
        U_psi['meta']['resolution'] = []
        U_psi['meta']['j'] = [] # FIXME: for this one, the elements can be scalars or length 2 lists.
        
        # print("U:{}".format(U))
        # print(len(U['signal']))
        # print(U['signal'][0].shape)
        # r = 0
        print("len(U['signal']):{}".format(len(U['signal'])))
        for p1 in range(len(U['signal'])): # num iterations: number of nodes in the current layer that is being processed to give the next layer
            current_bw = U['meta']['bandwidth'][p1] * 2**path_margin
            #print("current_bw:{}".format(current_bw))
            #print("psi_xi:{}".format(psi_xi))
            psi_mask = current_bw > np.array(psi_xi) # REVIEW: I think this determines whether to continue on this path or not
            # In the paper, the scattering transform is computed only along frequency-decreasing paths
            #print(U)
            x_res = U['meta']['resolution'][p1]
            # scat_options['x_resolution'] = U['meta']['resolution'][p1]
            # scat_options['psi_mask'] = psi_mask
            # x_phi, x_psi, meta_phi, meta_psi = self._wavelet_1d(copy.deepcopy(U['signal'][p1]), copy.deepcopy(filters), copy.deepcopy(scat_options))
            x_phi, x_psi, meta_phi, meta_psi = self._wavelet_1d(copy.deepcopy(U['signal'][p1]), copy.deepcopy(filters), psi_mask=psi_mask, x_res=x_res)
            # U_phi['signal'][0, p1] = x_phi # FIXME: matlab version does U_phi.signal{1,p1}. This line might break
            # print("x_phi:{}".format(x_phi))
            U_phi['signal'].append(x_phi) # FIXME: matlab version does U_phi.signal{1,p1}. This line might break 
            # print("U['meta']:{}".format(U['meta']))
            # print("U_phi['meta']:{}".format(U_phi['meta']))
            # print("p1:{}".format(p1) )

            # so far, looks good.

            # U_phi['meta']['bandwidth'].append(U['meta']['bandwidth'][p1])
            # U_phi['meta']['resolution'].append(U['meta']['resolution'][p1])


            # if len(U['meta']['j']) > p1:
            #     U_meta_j = U['meta']['j'][p1]
            # else:
            #     U_meta_j = None
            # U_phi['meta']['j'].append(U_meta_j)

            if len(U['meta']['j']) > p1: # FIXME: this seems true unless the layer being processed is the first layer (root). Later change it to an if statement with the depth
                U_phi['meta']['j'].append(U['meta']['j'][p1]) # U['meta']['j'] is a rank 2 list and so U['meta']['j'][p1] is also a list
            U_phi['meta']['bandwidth'].append(meta_phi['bandwidth'])
            U_phi['meta']['resolution'].append(meta_phi['resolution'])

            U_psi['signal'] += [signal for idx, signal in enumerate(x_psi) if psi_mask[idx]] # FIXME: change names
            U_psi['meta']['bandwidth'] += [bw for idx, bw in enumerate(meta_psi['bandwidth']) if psi_mask[idx]]
            U_psi['meta']['resolution'] += [res for idx, res in enumerate(meta_psi['resolution']) if psi_mask[idx]]

            # FIXME: this seems true unless the layer being processed is the first layer (root). Later change it to an if statement with the depth
            if len(U['meta']['j']) > p1:
                U_meta_j = U['meta']['j'][p1]
                # U['meta']['j'] is a rank 2 list and so U['meta']['j'][p1] itself is a list
            else:
                U_meta_j = []
            
            U_psi['meta']['j'] += [U_meta_j + [meta_psi_j] for idx, meta_psi_j in enumerate(meta_psi['j']) if psi_mask[idx]] # FIXME: in the matlab version zeros(0,0), zeros(1,0) are used and its size can be measured. Not possible here. If the matlab version has [empty; empty; 1] for j at some p1 index, this will be [None, 1] according to this line, not [None, None, 1]. This might break...

        return U_phi, U_psi

    def _morlet_filter_bank_1d(self, layer):
        '''
        generates morlet filter bank

        inputs:
        -------
        - layer: int indicating layer number
        
        outputs:
        --------
        - filters: has following keys:
        'phi' has keys 'meta', 'filter'. 'meta' has keys 'k'
        'psi' has keys 'meta', 'filter'. 'meta' has keys 'k'. 
        The key 'filter' within keys 'phi' and 'psi' is a list of dict type objects where each dict has keys 'filter_len', 'coef' (and optionally 'start')
        'meta' has the following keys:
        n_filter_log, n_filter_lin, n_filter_octave, xi_psi, sigma_psi, sigma_phi, size_filter
        '''
        
        sigma0 = 2 / np.sqrt(3)
        
        filters = {}
        data_len = self._data_len
        # get filter parameters from the instance attributes at the given layer
        filters['meta'] = {}

        n_filter_log = self._n_filter_log[layer]
        n_filter_lin = self._n_filter_lin[layer]

        filters['meta']['n_filter_log'] = n_filter_log
        filters['meta']['n_filter_lin'] = n_filter_lin
        filters['meta']['n_filter_octave'] = self._n_filter_octave[layer]
        filters['meta']['xi_psi'] = self._xi_psi[layer]
        filters['meta']['sigma_psi'] = self._sigma_psi[layer]
        filters['meta']['sigma_phi'] = self._sigma_phi[layer]

        # The normalization factor for the wavelets, calculated using the filters
        # at the finest resolution (N)
        
        mode = self._mode
        if mode == 'symm':
            filter_len = 2 * data_len
        else:
            filter_len = data_len

        # adjust filter_len to a power of 2 that is just large enough to contain its data length for convolution
        filter_len = int(2**np.ceil(np.log2(filter_len)))
        
        filters['meta']['size_filter'] = filter_len
        psi_center, psi_bw, phi_bw = self._morlet_freq_1d(filters['meta'])
        
        psi_sigma = sigma0 * np.pi / 2. / psi_bw
        phi_sigma = sigma0 * np.pi / 2. / phi_bw
        
        # Calculate normalization of filters so that sum of squares does not
        # exceed 2. This guarantees that the scattering transform is
        # contractive.
        # REVIEW: why 2? also, doesn't the modulus operator already contract it? why is this necessary?
        sum_sqs = np.zeros(filter_len)
        
        # As it occupies a larger portion of the spectrum, it is more
        # important for the logarithmic portion of the filter bank to be
        # properly normalized, so we only sum their contributions.
        for j1 in range(n_filter_log):
            tmp = self._gabor(filter_len, psi_center[j1], psi_sigma[j1])
            tmp = self._morletify(tmp, psi_sigma[j1])
            sum_sqs = sum_sqs + np.abs(tmp)**2
        
        psi_amp = np.sqrt(2 / max(sum_sqs))
        
        #print(filters['psi']['filter'])
        # Apply the normalization factor to the filters.
        filters['psi'] = {'meta':{'k':[]}, 'filter':[]}
        for j1 in range(n_filter_log + n_filter_lin):
            tmp = self._gabor(filter_len, psi_center[j1], psi_sigma[j1])
            tmp = self._morletify(tmp,psi_sigma[j1])
            filter_j = self._optimize_filter(psi_amp * tmp)
            filters['psi']['filter'].append(filter_j)
            filters['psi']['meta']['k'].append(j1)
        # FIXME: meaning of j1 and k?

        # Calculate the associated low-pass filter
        filters['phi'] = {'filter':self._gabor(filter_len, 0, phi_sigma)}
        filters['phi']['filter'] = self._optimize_filter(filters['phi']['filter'])
        filters['phi']['meta'] = {'k':[n_filter_log + n_filter_lin]}

        return filters

    def _gabor(self, filter_len, xi, sigma):
        '''
        inputs:
        -------
        - filter_len: int type
        - xi: float
        - sigma: float

        outputs:
        --------
        - f: np.array shaped (filter_len,)
        '''
        # NOTE: this function has been manually confirmed with a few inputs that the results are identical to that of the matlab version
        extent = 1 # extent of periodization - the higher, the better
        sigma = 1 / sigma
        f = np.zeros(filter_len)
        
        # calculate the 2*pi-periodization of the filter over 0 to 2*pi*(filter_len-1)/filter_len
        for k in range(-extent, 2 + extent):
            f += np.exp(-((np.arange(filter_len) - k * filter_len) / filter_len * 2 * np.pi - xi)**2. / (2 * sigma**2))
        return f

    def _morletify(self, f, sigma):
        '''
        inputs:
        -------
        - f: np.array
        - sigma: float

        outputs:
        --------
        - f: rank 1 np.array with shape identical to that of f

        '''
        f0 = f[0]
        f = f - f0 * self._gabor(len(f), 0, sigma)
        return f

    def _truncate_filter(self, filter, thresh=1e-3):
        '''
        truncates the given fourier transform of the filter. this filter will have values that are high in
        only a small region. in this case, one can make a similar filter which is a hard-thresholded version
        of the given filter. this can siginificantly speedup the computation as the convolution in the fourier
        domain is a multiplication and so only the nonzero values have to be considered.
        
        first, the smallest region that contains all the values that are above a certain threshold is identified.
        this region is extended slightly so that the region length is N/2^m where N is the length of the given
        filter and m is some integer. during this process, the adjusted region's start and end point might be 
        beyond the given filter. In this case, the filter is wrapped around (periodically extend the given filter
        and then take the adjusted region. 
        FIXME: what is the basis of the idea of wrapping around?
        
        in order to use this filter and later reconstruct to the correct original size, keys named
        'start' and 'filter_len' are stored in the output dictionary

        inputs:
        -------
        - filter: rank 1 array which is the fourier representation of the filter
        - thresh: threshold relative to the maximum value of the given filter's absolute values in fourier domain, between 0 and 1

        outputs:
        --------
        - filter_truncated: dict type object containing the following keys:
        coef: the truncated filter
        start: starting index of the fourier domain support
        filter_len: original length of the filter
        '''
        # print(filter)
        filter_len = len(filter)
        
        filter_truncated = {'filter_len':filter_len}

        # FIXME: for consistency the max() function in matlab is implemented below.
        # after running tests, consider replacing with idx_max = np.argmax(filter)
        maxabs = np.abs(filter).max()
        maxangle = np.angle(filter[np.abs(filter) == maxabs]).max()
        idx_max = np.where(np.logical_and(np.abs(filter) == maxabs, np.angle(filter) == maxangle))[0][0]

        filter = np.roll(filter, int(filter_len / 2) - (idx_max + 1))
        # np.where()'s return type is a tuple and so access the 0th index
        idx = np.where(np.abs(filter) > (np.abs(filter).max() * thresh))[0] 
        
        idx1 = idx[0]
        idx2 = idx[-1]

        nonzero_len = idx2 - idx1 + 1
        nonzero_len = int(np.round(filter_len / 2**(np.floor(np.log2(filter_len / nonzero_len)))))

        # before np.round(), add small amount since in np.round(), halfway values are 
        # rounded to the nearest even value, i.e., np.round(2.5) gives 2.0, NOT 3
        # if the amount is too small (1e-17, for example), does not work
        idx1 = int(np.round(np.round((idx1 + idx2) / 2 + 1e-6) - nonzero_len / 2 + 1e-6))
        idx2 = idx1 + int(nonzero_len) - 1
        
        filter = filter[np.arange(idx1, idx2 + 1) % filter_len]

        filter_truncated['coef'] = filter
        filter_truncated['start'] = int(idx1 - (filter_len / 2 - idx_max) + 1)

        return filter_truncated

