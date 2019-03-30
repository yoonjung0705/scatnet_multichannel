'''
test functions for scatnet.py. When function return values are tested, 
results are compared with that of matlab functions. If function returns a list of
rank 1 arrays, arrays are concatenated before comparison
NOTE: when an array is stored in a .mat file using save() in matlab, the data is retrieved using
f = h5py.File(filename)
If in matlab the array size is described as (data_len, n_data), when you read the data through python,
the size is (n_data, data_len). This holds even when n_data is 1, meaning, if you generate a column
in matlab, the shape when read in python nparray is (1, data_len), NOT (data_len,)

Therefore, when writing test functions extra care should be taken for functions that have arguments type
lists or arrays

FIXME: check if upon function call the arguments do not change (DONE so far)
'''
import os
import unittest
import numpy as np
import glob
import re
import scatnet as sn
import h5py
import copy


TEST_DATA_FILEPATH = '/home/yoonjung/repos/python/scatnet_multichannel/matlab_test_data/'

class ScatnetTestCase(unittest.TestCase):
    def test_T_to_J(self):
        '''
        - test if J is list when audio and scalar when dyadic
        - test if argument filt_opt remains instact upon function call
        - test if input argument does not change upon function call
        FIXME: check if Q,J,B are altogether list or scalar at the same time?
        NOTE: J can be negative if T too small
        '''
        for T in [10, 100, 1000, 10000]:
            s = sn.default_filter_opt('audio', 5)
            s_cp = copy.deepcopy(s)
            J = sn.T_to_J(T, s)
            self.assertIsInstance(J, list)
            self.assertEqual(s, s_cp)

            s = sn.default_filter_opt('dyadic', 5)
            s_cp = copy.deepcopy(s)
            J = sn.T_to_J(T, s)
            self.assertIsInstance(J, (int, float))
            self.assertEqual(s, s_cp)

        s = sn.default_filter_opt('audio', 10)
        s_cp = copy.deepcopy(s)
        J = sn.T_to_J(T, s)
        self.assertEqual(s, s_cp)

    def test_default_filter_opt(self):
        '''
        - test if Q and J are present as keys
        - test if Q and J are either length 2 list or number
        - test if J values are equal to that of the matlab function
        '''
        # self.assertRaises(sn.default_filter_opt('image', 10))
        for filter_type in ['audio', 'dyadic']:
            s = sn.default_filter_opt(filter_type, 5)
            self.assertIn('Q', s.keys())
            self.assertIn('J', s.keys())
            if isinstance(s['Q'], list):
                self.assertEqual(len(s['Q']), 2)
            else:
                self.assertTrue(s['Q'] > 0)

            if isinstance(s['J'], list):
                self.assertEqual(len(s['J']), 2) # FIXME: check if J field being list means length being 2
            else:
                self.assertTrue(s['J'] > 0)
        
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'default_filter_options'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.csv')
        regex = matlab_fun + '\{([a-z]+)\}\{([0-9]+)\}' + '.csv'
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            filter_type = match.group(1)
            avg_len = int(match.group(2))

            results = np.array(sn.default_filter_opt(filter_type, avg_len)['J'])
            ref_results = np.genfromtxt(test_file, delimiter=',')
            self.assertTrue(np.isclose(results, ref_results, rtol=1e-5, atol=1e-8).all())


    def test_fill_struct(self):
        '''
        - test if key value pair added when key not present
        - test if key value pair not added when key present
        '''
        s = {'key1':1, 'key2':3}
        sn.fill_struct(s, key1=1)
        sn.fill_struct(s, key2=2)
        sn.fill_struct(s, key3=3)
        self.assertEqual(s['key1'], 1)
        self.assertEqual(s['key2'], 3)
        self.assertEqual(s['key3'], 3)

    def test_morlet_freq_1d(self):
        '''
        - test if xi_psi, bw_psi are both type list.
        - test if bw_phi and all elements in bw_psi elements are positive. 
        NOTE: xi_psi can have negative elements. The stepsize in xi_psi in the linearly spaced
        spectrum can in theory be negative if sigma_phi is small although for parameters that
        construct phi this does not happen. However, even though the stepsize is positive for normal
        inputs, the number of steps taken linearly towards the negative frequency regime can result
        in negative values of center frequencies
        
        REVIEW: confirm whether the parameters that result in negative center frequencies are feasible
        If not, no need to test for cases having negative center frequencies  

        - test if xi_psi, bw_psi have length J+P, J+P+1, respectively
        - test if filt_opt does not change upon function call
        FIXME: add test case where output lists have length 0(?) or 1
        '''
        filt_opt = {'xi_psi':0.5, 'sigma_psi':0.4, 'sigma_phi':0.5, 'J':11,
            'Q':8, 'P':5, 'phi_dirac': True}
        # retain a copy of filt_opt to confirm no change upon function call
        filt_opt_cp = filt_opt.copy() 
        xi_psi, bw_psi, bw_phi = sn.morlet_freq_1d(filt_opt)
        self.assertIsInstance(xi_psi, list)
        self.assertIsInstance(bw_psi, list)
        # self.assertTrue(all([xi > 0 for xi in xi_psi]))
        self.assertTrue(all([bw > 0 for bw in bw_psi]))
        self.assertTrue(bw_phi > 0)
        self.assertEqual(len(xi_psi), filt_opt['J'] + filt_opt['P'])
        self.assertEqual(len(bw_psi), filt_opt['J'] + filt_opt['P'] + 1)
        self.assertEqual(filt_opt, filt_opt_cp)

        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'morlet_freq_1d'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        regex = matlab_fun + '\{([0-9]+\.?[0-9]*)\}' * 7 + '.mat'
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            Q = int(match.group(1))
            J = int(match.group(2))
            P = int(match.group(3))
            xi_psi = float(match.group(4))
            sigma_psi = float(match.group(5))
            sigma_phi = float(match.group(6))
            phi_dirac = bool(int(match.group(7)))

            options = {}
            options = sn.fill_struct(options, Q=Q, J=J, P=P, xi_psi=xi_psi, sigma_psi=sigma_psi, sigma_phi=sigma_phi, phi_dirac=phi_dirac)
            options_orig = copy.deepcopy(options)
            xi_psi, bw_psi, bw_phi = sn.morlet_freq_1d(options)

            xi_psi = np.array(xi_psi)
            bw_psi = np.array(bw_psi)
            bw_phi = np.array(bw_phi)

            ref_results_file = h5py.File(test_file)
            xi_psi_ref = np.array(ref_results_file['xi_psi']).squeeze(axis=1)
            bw_psi_ref = np.array(ref_results_file['bw_psi']).squeeze(axis=1)
            bw_phi_ref = np.array(ref_results_file['bw_phi']).squeeze(axis=1)[0]

            # xi_psi, xi_psi_ref, bw_psi, bw_psi_ref are all rank 1 arrays. bw_phi, bw_phi_ref are both float type scalars
            self.assertTrue(np.isclose(xi_psi, xi_psi_ref, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(bw_psi, bw_psi_ref, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(bw_phi, bw_phi_ref, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertEqual(options, options_orig)

    def test_pad_signal(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'pad_signal'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        regex = matlab_fun + '\{([0-9]+)\}' * 3 + '\{([a-z]+)\}' + '\{([0-9])\}' + '.mat'
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            data_len = int(match.group(1))
            n_data = int(match.group(2))
            pad_len = int(match.group(3))
            mode = str(match.group(4))
            center = bool(int(match.group(5)))

            ref_results_file = h5py.File(test_file)
            data_in_ref = np.array(ref_results_file['data_in'])
            data_out_ref = np.array(ref_results_file['data_out'])
            data_in_ref_orig = np.copy(data_in_ref)

            data_out = sn.pad_signal(data_in_ref, pad_len=pad_len, mode=mode, center=center)
            self.assertTrue(np.isclose(data_out, data_out_ref, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(data_in_ref, data_in_ref_orig, rtol=1e-5, atol=1e-8).all())    

    def test_unpad_signal(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'unpad_signal'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        regex = matlab_fun + '\{([0-9]+)\}' * 5 + '.mat'
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            data_len = int(match.group(1))
            n_data = int(match.group(2))
            res = int(match.group(3))
            orig_len = int(match.group(4))
            center = bool(int(match.group(5)))

            ref_results_file = h5py.File(test_file)
            data_in_ref = np.array(ref_results_file['data_in'])
            data_out_ref = np.array(ref_results_file['data_out'])
            data_in_ref_orig = np.copy(data_in_ref)

            data_out = sn.unpad_signal(data_in_ref, res=res, orig_len=orig_len, center=center)
            self.assertTrue(np.isclose(data_out, data_out_ref, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(data_in_ref, data_in_ref_orig, rtol=1e-5, atol=1e-8).all())               

    def test_periodize_filter(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'periodize_filter'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        regex = matlab_fun + '\{([0-9]+)\}' + '.mat'
        for test_file in test_files:
            ref_results_file = h5py.File(test_file)
            # the values of ref_results_file have both shape (1, filter_len)
            filter_in_ref = np.array(ref_results_file['filter_f'])[0]
            coef_out_ref = np.array(ref_results_file['coef_concat'])[0]
            filter_in_ref_orig = np.copy(filter_in_ref)

            coef_out = np.concatenate(sn.periodize_filter(filter_in_ref)['coef'], axis=0)
            # coef_out has shape (filter_len,)
            self.assertTrue(np.isclose(coef_out, coef_out_ref, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(filter_in_ref, filter_in_ref_orig, rtol=1e-5, atol=1e-8).all())            

    def test_conv_sub_1d_filt_array(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        NOTE: when reading arrays with complex numbers from matlab, the format for each number becomes a tuple.
        to avoid this, I saved the real and imaginary part separately and compare the results for real and imaginary
        separately.
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'conv_sub_1d'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        # filt argument given as nparray
        regex = matlab_fun + '\{([0-9]+)\}' * 4 + '.mat' 
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            if match is None:
                continue
            data_len = int(match.group(1))
            n_data = int(match.group(2))
            filt_len = int(match.group(3))
            ds = int(match.group(4))

            ref_results_file = h5py.File(test_file)
            data_in_ref_real = np.array(ref_results_file['data_in_real'])
            data_in_ref_imag = np.array(ref_results_file['data_in_imag'])
            # ref_results_file['filt_in'] has size (1, filt_len)
            filt_in_ref_real = np.array(ref_results_file['filt_in_real'])[0]
            filt_in_ref_imag = np.array(ref_results_file['filt_in_imag'])[0]
            data_out_ref_real = np.array(ref_results_file['data_out_real'])
            data_out_ref_imag = np.array(ref_results_file['data_out_imag'])

            data_in_ref = data_in_ref_real + data_in_ref_imag * 1j
            filt_in_ref = filt_in_ref_real + filt_in_ref_imag * 1j
            data_in_ref_orig = np.copy(data_in_ref)
            filt_in_ref_orig = np.copy(filt_in_ref)
            data_out = sn.conv_sub_1d(data_in_ref, filt_in_ref, ds)
            data_out_real = np.real(data_out)
            data_out_imag = np.imag(data_out)

            self.assertTrue(np.isclose(data_out_real, data_out_ref_real, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(data_out_imag, data_out_ref_imag, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(filt_in_ref, filt_in_ref_orig, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(data_in_ref, data_in_ref_orig, rtol=1e-5, atol=1e-8).all())


    def test_conv_sub_1d_fourier_multires(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        NOTE: when reading arrays with complex numbers from matlab, the format for each number becomes a tuple.
        to avoid this, I saved the real and imaginary part separately and compare the results for real and imaginary
        separately.
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'conv_sub_1d'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        # filt argument given as nparray
        regex = matlab_fun + '\{([0-9]+)\}' * 2 + '\{fourier_multires\}'  + '\{([0-9]+)\}' * 2 + '.mat' 
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            if match is None:
                continue
            data_len = int(match.group(1))
            n_data = int(match.group(2))
            N = int(match.group(3))
            ds = int(match.group(4))

            ref_results_file = h5py.File(test_file)
            data_in_ref_real = np.array(ref_results_file['data_in_real'])
            data_in_ref_imag = np.array(ref_results_file['data_in_imag'])
            data_in_ref = data_in_ref_real + data_in_ref_imag * 1j
            data_in_ref_orig = np.copy(data_in_ref)            
           
            coef_in_ref_real = np.array(ref_results_file['coef_real'])
            coef_in_ref_imag = np.array(ref_results_file['coef_imag'])
            coef_in_ref = coef_in_ref_real + coef_in_ref_imag * 1j
            # ref_results_file['filt_in'] has size (1, filt_len)

            # reconstruct filt['coef']
            filt = {'N':N, 'type':'fourier_multires'}
            coef = []
            n = float(N)
            while n.is_integer():
                n = int(n)
                coef.append(coef_in_ref[:n])
                coef_in_ref = np.delete(coef_in_ref, np.s_[:n])
                n = float(n) / 2
            filt['coef'] = coef
            filt_orig = copy.deepcopy(filt)

            data_out_ref_real = np.array(ref_results_file['data_out_real'])
            data_out_ref_imag = np.array(ref_results_file['data_out_imag'])

            data_out = sn.conv_sub_1d(data_in_ref, filt, ds)
            data_out_real = np.real(data_out)
            data_out_imag = np.imag(data_out)

            self.assertTrue(np.isclose(data_out_real, data_out_ref_real, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(data_out_imag, data_out_ref_imag, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(data_in_ref_orig, data_in_ref, rtol=1e-5, atol=1e-8).all())
            for key in ['N', 'type']:
                self.assertEqual(filt[key], filt_orig[key])
            for idx in range(len(filt_orig['coef'])):
                self.assertTrue(np.isclose(filt['coef'][idx], filt_orig['coef'][idx], rtol=1e-5, atol=1e-8).all())


    def test_conv_sub_1d_fourier_truncated(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        NOTE: when reading arrays with complex numbers from matlab, the format for each number becomes a tuple.
        to avoid this, I saved the real and imaginary part separately and compare the results for real and imaginary
        separately.
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'conv_sub_1d'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        # filt argument given as nparray
        regex = matlab_fun + '\{([0-9]+)\}' * 2 + '\{fourier_truncated\}'  + '\{([0-9]+)\}' * 3 + '\{(-?[0-9]+)\}' + '\{([0-9]+\.?[0-9]*)\}' + '.mat' 
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            if match is None:
                continue

            data_len = int(match.group(1))
            n_data = int(match.group(2))
            filt_len = int(match.group(3))
            ds = int(match.group(4))
            N = int(match.group(5))
            start = int(match.group(6))
            thresh = float(match.group(7))

            ref_results_file = h5py.File(test_file)
            data_in_ref_real = np.array(ref_results_file['data_in_real'])
            data_in_ref_imag = np.array(ref_results_file['data_in_imag'])
            data_in_ref = data_in_ref_real + data_in_ref_imag * 1j
            data_in_ref_orig = np.copy(data_in_ref)            
            # coef_real and coef_imag have shapes (1, filt_len)
            coef_in_ref_real = np.array(ref_results_file['coef_real'])[0]
            coef_in_ref_imag = np.array(ref_results_file['coef_imag'])[0]
            coef_in_ref = coef_in_ref_real + coef_in_ref_imag * 1j

            # reconstruct filt['coef']
            filt = {'N':N, 'type':'fourier_truncated', 'start':start - 1, 'coef':coef_in_ref, 'recenter':True}
            filt_orig = copy.deepcopy(filt)
            data_out_ref_real = np.array(ref_results_file['data_out_real'])
            data_out_ref_imag = np.array(ref_results_file['data_out_imag'])

            data_out = sn.conv_sub_1d(data_in_ref, filt, ds)
            data_out_real = np.real(data_out)
            data_out_imag = np.imag(data_out)

            self.assertTrue(np.isclose(data_out_real, data_out_ref_real, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(data_out_imag, data_out_ref_imag, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(data_in_ref_orig, data_in_ref, rtol=1e-5, atol=1e-8).all())
            for key in ['N', 'type', 'start', 'recenter']:
                self.assertEqual(filt[key], filt_orig[key])
            self.assertTrue(np.isclose(filt['coef'], filt_orig['coef'], rtol=1e-5, atol=1e-8).all())


    '''
    matlab: 
    data = rand(16,5);
    coef = rand(16,1);
    filt = truncate_filter(coef, 0.8, 1);
    conv_sub_1d(data, filt, 2); # size is (4,5)

    python:
    data = np.random.random((5,16))
    coef = np.random.random((16,))
    filt = sn.truncate_filter(coef, 0.8)
    sn.conv_sub_1d(data, filt, 2) # shape is (5,4) 


    '''

    def test_truncate_filter(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        NOTE: when reading arrays with complex numbers from matlab, the format for each number becomes a tuple.
        to avoid this, I saved the real and imaginary part separately and compare the results for real and imaginary
        separately.
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'truncate_filter'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        # filt argument given as nparray
        regex = matlab_fun + '\{([0-9]+)\}' + '\{([0-9]+\.?[0-9]*)\}' + '.mat' 
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            if match is None:
                continue

            filt_len = int(match.group(1))
            thresh = float(match.group(2))

            ref_results_file = h5py.File(test_file)
            start_ref = ref_results_file['start']
            N_ref = ref_results_file['N']
            recenter_ref = ref_results_file['recenter']
            type_ref = ref_results_file['type']
            # the following arrays have shapes (1, coef_len)
            coef_in_ref_real = np.array(ref_results_file['coef_in_real'])[0]
            coef_in_ref_imag = np.array(ref_results_file['coef_in_imag'])[0]
            coef_out_ref_real = np.array(ref_results_file['coef_out_real'])[0]
            coef_out_ref_imag = np.array(ref_results_file['coef_out_imag'])[0]

            coef_in_ref = coef_in_ref_real + coef_in_ref_imag * 1j
            coef_in_ref_orig = np.copy(coef_in_ref)

            filt = sn.truncate_filter(coef_in_ref, thresh)

            self.assertTrue(np.isclose(np.real(filt['coef']), coef_out_ref_real, rtol=1e-5, atol=1e-8).all())
            self.assertTrue(np.isclose(np.imag(filt['coef']), coef_out_ref_imag, rtol=1e-5, atol=1e-8).all())
            self.assertEqual(filt['start'] + 1, int(start_ref[0,0]))
            self.assertEqual(filt['N'], N_ref[0,0])
            self.assertTrue(filt['recenter'])
            self.assertEqual(filt['type'], 'fourier_truncated')
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(coef_in_ref_orig, coef_in_ref, rtol=1e-5, atol=1e-8).all())

    def test_gabor(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        NOTE: when reading arrays with complex numbers from matlab, the format for each number becomes a tuple.
        to avoid this, I saved the real and imaginary part separately and compare the results for real and imaginary
        separately.
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'gabor'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        # filt argument given as nparray
        regex = matlab_fun + '\{([0-9]+)\}' + '\{(-?[0-9]+\.?[0-9]*)\}' * 2 + '.mat' 
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            if match is None:
                continue

            N = int(match.group(1))
            xi = float(match.group(2))
            sigma = float(match.group(3))

            ref_results_file = h5py.File(test_file)
            # the following array has shape (1, len)
            data_out_ref = ref_results_file['data_out'][0]
            data_out = sn.gabor(N, xi, sigma)

            self.assertTrue(np.isclose(data_out, data_out_ref, rtol=1e-5, atol=1e-8).all())

    def test_morletify(self):
        '''FIXME: add tests on python only (not comparing with matlab) to test sizes, other stuff
        NOTE: when reading arrays with complex numbers from matlab, the format for each number becomes a tuple.
        to avoid this, I saved the real and imaginary part separately and compare the results for real and imaginary
        separately.
        - test if input argument does not change upon function call
        '''
        # calculate fields using python function using parameters retrieved from matlab test data file names
        matlab_fun = 'morletify'
        test_files = glob.glob(TEST_DATA_FILEPATH + matlab_fun + '*.mat')
        # filt argument given as nparray
        regex = matlab_fun + '\{([0-9]+)\}' + '\{(-?[0-9]+\.?[0-9]*)\}' * 3 + '.mat' 
        for test_file in test_files:
            match = re.search(regex, os.path.basename(test_file))
            if match is None:
                continue

            N = int(match.group(1))
            xi = float(match.group(2))
            sigma = float(match.group(3))
            psi_sigma = float(match.group(4))

            ref_results_file = h5py.File(test_file)
            # the following arrays have shape (1, len)
            data_in_ref = ref_results_file['data_in'][0]
            data_out_ref = ref_results_file['data_out'][0]
            data_in_ref_orig = np.copy(data_in_ref) # deepcopy
            data_out = sn.morletify(data_in_ref, psi_sigma)

            self.assertTrue(np.isclose(data_out, data_out_ref, rtol=1e-5, atol=1e-8).all())
            # check if input array does not change upon function call
            self.assertTrue(np.isclose(data_in_ref_orig, data_in_ref, rtol=1e-5, atol=1e-8).all())
    # FIXME: add tests on optimize_filter() and filter_freq()

    # def test_map_meta(self):
    #   '''
    #   - test if copying columns matches what is expected for the following cases:
    #   1 column to 1 column
    #   3 columns to 3 columns
    #   2 columns to 4 columns
        
    #   when from_meta and to_meta have an overlap of key value pairs while also having
    #   nonoverlapping key value pairs
        
    #   when to_meta has no key value pairs
    #   when to_meta has no overlapping key value pairs
    #   when from_meta has no key value pairs
    #   2 columns to 1 column (should raise error)
    #   when index is out of bound for from_meta (should raise error)
    #   when index is out of bound for to_meta
    #   when from_meta and to_meta have an overlap of key value pairs while
    #   a) to_meta having empty list of indices
    #   b) from_meta having empty list of indices
    #   when "exclude" argument is not empty
    #   from_meta is intact after function call

    #   FIXME: for shared keys, if to_ind goes out of bound, should to_meta's shared key be
    #   extended to incorporate that? or should it raise an error? Current version does not extend
    #   '''
    #   # 1 column to 1 column
    #   from_ind = 2
    #   to_ind = 3
    #   from_meta = self.create_meta(('key1', (8,5)), ('key2', (10,5)))
    #   to_meta = self.create_meta(('key1', (4,5)), ('key3', (4,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if to_meta's shared key's to_ind values are identical to 
    #   # from_meta's shared key's from_ind values 
    #   self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's shared key's indices NOT IN to_ind are identical to 
    #   # to_meta_orig's shared key's indices NOT IN to_ind 
    #   to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx != to_ind]
    #   self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
    #       to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key3'], 
    #       to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
    #       to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

    #   # 3 columns to 3 columns
    #   from_ind = [2,1,0]
    #   to_ind = [1,3,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (6,5)), ('key3', (8,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if to_meta's shared key's to_ind values are identical to 
    #   # from_meta's shared key's from_ind values 
    #   self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's shared key's indices NOT IN to_ind are identical to 
    #   # to_meta_orig's shared key's indices NOT IN to_ind 
    #   to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
    #   self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
    #       to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key3'], 
    #       to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
    #       to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

    #   # 2 columns to 4 columns
    #   from_ind = [2,1]
    #   to_ind = [1,3,2,0]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (6,5)), ('key3', (8,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if to_meta's shared key's to_ind values are identical to 
    #   # from_meta's shared key's from_ind values 
    #   self.assertTrue(np.isclose(np.tile(from_meta['key1'][from_ind], 
    #       (int(len(to_ind)/len(from_ind)), 1)), 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's shared key's indices NOT IN to_ind are identical to 
    #   # to_meta_orig's shared key's indices NOT IN to_ind 
    #   to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
    #   self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
    #       to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key3'], 
    #       to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(np.tile(from_meta['key2'][from_ind], 
    #       (int(len(to_ind)/len(from_ind)), 1)),
    #       to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())


    #   # when to_meta has no key value pairs, try 3 cols to 3 cols
    #   from_ind = [2,1,0]
    #   to_ind = [1,3,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = {}
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
    #       to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

    #   # when there are no shared keys, 3 columns to 3 columns
    #   from_ind = [2,1,0]
    #   to_ind = [1,3,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key3', (6,5)), ('key4', (8,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key3'], 
    #       to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
    #   self.assertTrue(np.isclose(to_meta_orig['key4'], 
    #       to_meta_out['key4'], rtol=1e-5, atol=1e-8).all())
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
    #       to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

    #   # when from_meta has no key value pairs, try 1 col to 1 col
    #   from_ind = 2
    #   to_ind = 3
    #   from_meta = {}
    #   to_meta = self.create_meta(('key1', (4,5)), ('key2', (4,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key1'], 
    #       to_meta_out['key1'], rtol=1e-5, atol=1e-8).all())
    #   self.assertTrue(np.isclose(to_meta_orig['key2'], 
    #       to_meta_out['key2'], rtol=1e-5, atol=1e-8).all())

    #   # 2 columns to 1 column (should raise error)
    #   from_ind = [2,1]
    #   to_ind = 1
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
    #   to_meta_orig = to_meta.copy()
    #   # following should raise error
    #   with self.assertRaises(ValueError):
    #       to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)

    #   # when index is out of bound for from_meta (should raise error)
    #   from_ind = [20,1]
    #   to_ind = [1,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
    #   to_meta_orig = to_meta.copy()
    #   # following should raise error
    #   with self.assertRaises(IndexError):
    #       to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)

    #   # when index is out of bound for to_meta
    #   from_ind = [3,1]
    #   to_ind = [20,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   # check if to_meta's shared key's to_ind values are identical to 
    #   # from_meta's shared key's from_ind values 
    #   self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's shared key's indices NOT IN to_ind are identical to 
    #   # to_meta_orig's shared key's indices NOT IN to_ind 
    #   to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
    #   self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
    #       to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key3'], 
    #       to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
    #       to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

    #   # when from_meta having empty list of indices, check if there's no change in to_meta
    #   from_ind = []
    #   to_ind = [1,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)     
    #   # following should return to_meta without any change
    #   for key in to_meta.keys():
    #       self.assertTrue(np.isclose(to_meta_orig[key], to_meta_orig[key], 
    #           rtol=1e-5, atol=1e-8).all())
            
    #   # when to_meta having empty list of indices, check if there's no change in to_meta
    #   from_ind = [1,2]
    #   to_ind = []
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
    #   to_meta_orig = to_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)     
    #   # following should return to_meta without any change
    #   for key in to_meta.keys():
    #       self.assertTrue(np.isclose(to_meta_orig[key], to_meta_orig[key], 
    #           rtol=1e-5, atol=1e-8).all())

    #   # when "exclude" argument is not empty
    #   from_ind = [2,1,0]
    #   to_ind = [1,3,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)), ('key3', (4,5)))
    #   to_meta = self.create_meta(('key1', (6,5)), ('key2', (8,5)), ('key4', (16,5)))
    #   to_meta_orig = to_meta.copy()
    #   exclude = ['key2']
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=exclude)
    #   # check if to_meta's shared key's to_ind values are identical to 
    #   # from_meta's shared key's from_ind values 
    #   self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
    #       to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's shared key's indices NOT IN to_ind are identical to 
    #   # to_meta_orig's shared key's indices NOT IN to_ind 
    #   to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
    #   self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
    #       to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
    #   # check if to_meta's pure key's values are identical to 
    #   # to_meta_orig's pure key's values
    #   self.assertTrue(np.isclose(to_meta_orig['key4'], 
    #       to_meta_out['key4'], rtol=1e-5, atol=1e-8).all())
    #   # check if from_meta's pure key's values are copied into to_meta
    #   self.assertTrue(np.isclose(from_meta['key3'][from_ind], 
    #       to_meta_out['key3'][to_ind], rtol=1e-5, atol=1e-8).all())
    #   # check if key in exclude list is not affected in to_meta
    #   self.assertTrue(np.isclose(to_meta_orig['key2'], 
    #       to_meta_out['key2'], rtol=1e-5, atol=1e-8).all())

    #   # from_meta is intact after function call
    #   from_ind = [2,1,0]
    #   to_ind = [1,3,2]
    #   from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
    #   to_meta = self.create_meta(('key1', (6,5)), ('key3', (8,5)))
    #   from_meta_orig = from_meta.copy()
    #   to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)
    #   for key in from_meta.keys():
    #       self.assertTrue(np.isclose(from_meta_orig[key], 
    #           from_meta[key], rtol=1e-5, atol=1e-8).all())

    #   # when shape[1] is different (should raise error)
    #   # 1 column to 1 column
    #   from_ind = 2
    #   to_ind = 3
    #   from_meta = self.create_meta(('key1', (8,7)), ('key2', (10,3)))
    #   to_meta = self.create_meta(('key1', (4,5)), ('key3', (4,5)))
    #   to_meta_orig = to_meta.copy()

    #   with self.assertRaises(ValueError):
    #       to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind)

    # def create_meta(self, *args):
    #   '''
    #   helper function for generating dict type objects used for testing map_meta()
    #   inputs:
    #   -------
    #   - args: length 2 tuple describing a single key-value pair to be added to the dict
    #   example: create_meta(('key1', (7,5)), ('key2', (12,5)), ('key3', (3,5)))
    #   will create a dict type object with keys being key1, key2, key3, and the corresponding
    #   values being np arrays with shape (7,5), (12,5), (3,5), respectively
    #   '''
    #   meta = {}
    #   for kv in args:
    #       key, shape = kv
    #       meta[key] = np.random.random(shape)
    #   return meta 





if __name__ == '__main__':
    unittest.main()
