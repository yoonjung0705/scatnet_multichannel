'''module containing functions that are currently not used'''

def scat_features(x, params, avg_len, n_filter_octave=[1, 1], log_scat=True):
    '''returns feature matrix X from a set of time series using the scattering transform
    calculates the logarithm of the scattering transform and takes the mean along the time axis
    for the filter format and the boundary conditions, default parameters are used
    (filter_format='fourier_truncated', mode='symm')

    FIXME: seems not considering having multiple channels. consider deprecation

    inputs:
    -------
    x: rank d array whose last dimension corresponds to the time axis
    params: list of parameters. parameters can be either lists or 1d arrays
    avg_len: scaling function width for scattering transform
    n_filter_octave: number of filters per octave for scattering transform
    log_scat: boolean indicating whether to take the log of the scattering transform results
    
    outputs:
    --------
    X: rank 2 array sized (n_timeseries, n_channels * n_nodes * data_len)
    y: rank 1 array sized (n_timeseries,) denoting the simulation parameters

    '''
    data_len = x.shape[-1]
    for idx, s in enumerate(x.shape[:-1]):
        assert(s == len(params[idx])), "Array shape does not comply with number of parameters"
    scat = ScatNet(data_len, avg_len, n_filter_octave=n_filter_octave)
    S = scat.transform(x)
    if log_scat: S = log_scat(S)
    S = stack_scat(S)
    #S = S(axis=-1)
    S = np.reshape(S, (-1, S.shape[-1]))
    
    return S

def merge_channels(x):
    '''returns an instance of the result of scat.transform() whose channels are merged into a single channel
    NOTE: subject to deprecation

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
