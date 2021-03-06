'''module that simulates stochastic trajectories'''
# TODO: check *_sample() functions to ensure simulations are correct
# TODO: consider adding gamma argument to one bead simulations
import os
import warnings
import numpy as np
import random
import torch
import glob
import re
from datetime import datetime
from itertools import product

import common_utils as cu

ROOT_DIR = './data/simulations/'

def sim_brownian(data_len, diff_coefs, dt, n_data=1, save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of brownian trajectories

    inputs:
    -------
    - data_len: int, length of each process
    - diff_coef: numeric or list or ndarray, diffusion coefficient. 
    - dt: float, time step between data points
    - n_data: int, number of processes in ensemble
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 1, data_len) which is an
        ensemble of brownian trajectories. the singleton dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are indices for the label values
        'label_names': list whose elements are string type, ['diff_coefs']
        'labels_lut': list where the values are the label values given the index values in 'labels'
        'dt': float type dt
    - (file_name): string type file name of the simulated data. returned if save_file is True
    '''
    concat_list = []
    if isinstance(diff_coefs, (int, float)):
        diff_coefs = [diff_coefs]
    diff_coefs = np.array(diff_coefs, dtype=dtype)
    n_diff_coefs = len(diff_coefs)
    file_size_est = data_len * len(diff_coefs) * n_data * np.dtype(dtype).itemsize
    file_size_est_gb = file_size_est / 1.e9
    if file_size_est_gb > 2.:
        warnings.warn("Generating file with size roughly {:.2f} GB".format(file_size_est_gb), Category=BytesWarning)

    for diff_coef in diff_coefs:
        increments = np.sqrt(2 * diff_coef * dt) * np.random.normal(0, 1, [n_data, data_len - 1])
        x0 = np.random.normal(0, 1, [n_data, 1])
        increments = np.concatenate([x0, increments], axis=1)
        processes = increments.cumsum(axis=1)
        concat_list.append(processes.astype(dtype))

    processes = np.stack(concat_list, axis=0)
    processes = np.expand_dims(processes, axis=-2)

    # reshape data
    n_data_total = n_diff_coefs * n_data
    processes = np.reshape(processes, (n_data_total, 1, data_len)) # shaped (n_data_total, 1, data_len)

    # reshape labels
    labels = [diff_coefs]
    labels = np.array(list(product(*labels)), dtype=dtype) # shaped (n_conditions, n_labels)
    labels_lut = [tuple(condition) for condition in labels]
    n_conditions = len(labels_lut)

    labels = np.arange(n_conditions) # shaped (n_conditions,)
    labels = np.repeat(labels, n_data) # shaped (n_conditions * n_samples_total,)                                                       

    samples = {'data':processes, 'labels':labels, 'label_names':['diff_coefs'], 'dt':dt, 'labels_lut':labels_lut}
    if not save_file:
        return samples

    nums = cu.match_filename(r'brw_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'brw_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_brownian_sample(data_len, diff_coefs, dt, n_data=1, save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of brownian trajectories for a given range of diffusion coefficients

    inputs:
    -------
    - data_len: int, length of each process
    - diff_coef: numeric or length 2 list-like representing low, high values of diffusion coefficients
    - dt: float, time step between data points
    - n_data: int, number of processes in ensemble
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 1, data_len) which is an
        ensemble of brownian trajectories. the singleton dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are the diff_coefs values
        'label_names': list whose elements are string type, ['diff_coefs']
        'dt': float type dt
    - (file_name): string type file name of the simulated data. returned if save_file is True
    '''
    if isinstance(diff_coefs, (int, float)):
        diff_coefs = np.array([diff_coefs, diff_coefs], dtype=dtype)
    assert(len(diff_coefs) == 2), "Invalid diff_coefs given: should be numeric or length 2 list-like format"
    diff_coef_low, diff_coef_high = diff_coefs
    diff_coef_samples = (diff_coef_high - diff_coef_low) * np.random.random(n_data,) + diff_coef_low 

    concat_list = []
    for diff_coef_sample in diff_coef_samples:
        process = sim_brownian(data_len, diff_coefs=diff_coef_sample, dt=dt, n_data=1, save_file=False, dtype=dtype)
        process = process['data']
        concat_list.append(process)
    processes = np.concatenate(concat_list, axis=0) # shaped (n_data, 1, data_len)

    samples = {'data':processes, 'labels':[diff_coef_samples], 'label_names':['diff_coefs'], 'dt':dt}
    if not save_file:
        return samples

    nums = cu.match_filename(r'brw_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'brw_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_poisson(data_len, lams, dt, n_data=1, save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of poisson processes

    inputs:
    -------
    - data_len: int, length of each process
    - lams: numeric or list or ndarray, expectation per interval
    - dt: time step between data points
    - n_data: number of processes in ensemble
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 1, data_len) which is an
        ensemble of poisson trajectories. the singleton dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are indices for the label values
        'label_names': list whose elements are string type, ['lams']
        'labels_lut': list where the values are the label values given the index values in 'labels'
        'dt': float type dt
    - (file_name): string type file name of the simulated data. data returned if save_file is True

    REVIEW: confirm this method of using fixed time step generates identical statistics to that of Gielespie algorithm
    '''
    if isinstance(lams, (int, float)):
        lams = [lams]
    lams = np.array(lams, dtype=dtype)
    n_lams = len(lams)
    
    file_size_est = data_len * n_lams * n_data * np.dtype(dtype).itemsize
    file_size_est_gb = file_size_est / 1.e9
    if file_size_est_gb > 2.:
        warnings.warn("Generating file with size roughly {:.2f} GB".format(file_size_est_gb), Category=BytesWarning)

    concat_list = []
    for lam in lams:
        increments = np.random.poisson(lam * dt, size=[n_data, data_len])
        processes = increments.cumsum(axis=1)
        concat_list.append(processes.astype(dtype))
    processes = np.stack(concat_list, axis=0)    
    processes = np.expand_dims(processes, axis=-2)

    # reshape data
    n_data_total = n_lams * n_data
    processes = np.reshape(processes, (n_data_total, 1, data_len)) # shaped (n_data_total, 1, data_len)

    # reshape labels
    labels = [lams]
    labels = np.array(list(product(*labels)), dtype=dtype) # shaped (n_conditions, n_labels)
    labels_lut = [tuple(condition) for condition in labels]
    n_conditions = len(labels_lut)

    labels = np.arange(n_conditions) # shaped (n_conditions,)
    labels = np.repeat(labels, n_data) # shaped (n_conditions * n_samples_total,)                                                       

    samples = {'data':processes, 'labels':labels, 'label_names':['lams'], 'dt':dt, 'labels_lut':labels_lut}
    if not save_file:
        return samples

    nums = cu.match_filename(r'pos_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'pos_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_poisson_sample(data_len, lams, dt, n_data=1, save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of poisson processes for a given range of lambda values

    inputs:
    -------
    - data_len: int, length of each process
    - lams: numeric or length 2 list-like representing low, high values of expectation per interval value
    - dt: time step between data points
    - n_data: int, number of processes in ensemble
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 1, data_len) which is an
        ensemble of poisson trajectories. the singleton dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are the lams values
        'label_names': list whose elements are string type, ['lams']
        'dt': float type dt
   - (file_name): string type file name of the simulated data. data returned if save_file is True

    REVIEW: confirm this method of using fixed time step generates identical statistics to that of Gielespie algorithm
    '''
    if isinstance(lams, (int, float)):
        lams = np.array([lams, lams], dtype=dtype)
    assert(len(lams) == 2), "Invalid lams given: should be numeric or length 2 list-like format"
    lam_low, lam_high = lams
    lam_samples = (lam_high - lam_low) * np.random.random(n_data,) + lam_low 

    concat_list = []
    for lam_sample in lam_samples:
        process = sim_poisson(data_len, lams=lam_sample, dt=dt, n_data=1, save_file=False, dtype=dtype)
        process = process['data']
        concat_list.append(process)
    processes = np.concatenate(concat_list, axis=0) # shaped (n_data, 1, data_len)

    samples = {'data':processes, 'labels':[lam_samples], 'label_names':['lams'], 'dt':dt}
    if not save_file:
        return samples

    nums = cu.match_filename(r'pos_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'pos_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_one_bead(data_len, ks, diff_coefs, dt, n_data=1, n_steps_initial=10000,
    save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of one bead simulation trajectories. as there is only one heat bath, this is a passive trajectory

    inputs:
    -------
    - data_len: int, length of each process
    - k: numeric or list or ndarray, spring constant
    - diff_coef: numeric or list or ndarray, diffusion coefficient
    - dt: time step between data points
    - n_data: number of processes in ensemble
    - n_steps_initial: number of steps to take in Langevin equation for simulating initial positions
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 1, data_len) which is an
        ensemble of one bead trajectories. the singleton dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are indices for the label values
        'label_names': list whose elements are string type, ['ks', 'diff_coefs']
        'labels_lut': list where the values are the label values given the index values in 'labels'
        'dt': float type dt
        'n_steps_initial': int type n_steps_initial
   - (file_name): string type file name of the simulated data. returned if save_file is True

    FIXME: check the code to see if actual simulation part is not mixed up with initial condition simulation
    '''
    if isinstance(ks, (int, float)):
        ks = [ks]
    if isinstance(diff_coefs, (int, float)):
        diff_coefs = [diff_coefs]

    ks = np.array(ks, dtype=dtype)
    diff_coefs = np.array(diff_coefs, dtype=dtype)
    n_ks = len(ks)
    n_diff_coefs = len(diff_coefs)

    file_size_est = data_len * n_diff_coefs * n_ks * n_data * np.dtype(dtype).itemsize
    file_size_est_gb = file_size_est / 1.e9
    if file_size_est_gb > 2.:
        warnings.warn("Generating file with size roughly {:.2f} GB".format(file_size_est_gb), Category=BytesWarning)

    processes = np.empty((n_ks, n_diff_coefs, n_data, data_len)).astype(dtype)

    for idx0, k in enumerate(ks):
        for idx1, diff_coef in enumerate(diff_coefs):
            prefactor1 = k * dt
            prefactor2 = np.sqrt(2 * diff_coef * dt)
            rand_nums = np.random.normal(0, 1, [n_steps_initial, n_data])
            x0 = np.zeros(n_data)
            for idx in range(n_steps_initial):
                x0 = x0 - prefactor1 * x0 + prefactor2 * rand_nums[idx]
            processes[idx0, idx1, :, 0] = x0

    for idx0, k in enumerate(ks):
        for idx1, diff_coef in enumerate(diff_coefs):
            x = processes[idx0, idx1, :, 0]
            prefactor1 = k * dt
            prefactor2 = np.sqrt(2 * diff_coef * dt)
            rand_nums = np.random.normal(0, 1, [data_len - 1, n_data])
            for idx in range(data_len - 1):
                x = x - prefactor1 * x + prefactor2 * rand_nums[idx]
                processes[idx0, idx1, :, idx + 1] = x

    processes = np.expand_dims(processes, axis=-2)

    # reshape data
    n_data_total = n_ks * n_diff_coefs * n_data
    processes = np.reshape(processes, (n_data_total, 1, data_len)) # shaped (n_data_total, 1, data_len)

    # reshape labels
    labels = [ks, diff_coefs]
    labels = np.array(list(product(*labels)), dtype=dtype) # shaped (n_conditions, n_labels)
    labels_lut = [tuple(condition) for condition in labels]
    n_conditions = len(labels_lut)

    labels = np.arange(n_conditions) # shaped (n_conditions,)
    labels = np.repeat(labels, n_data) # shaped (n_conditions * n_samples_total,)                                                       

    samples = {'data':processes, 'labels':labels, 'label_names':['ks', 'diff_coefs'], 'dt':dt, 'n_steps_initial':n_steps_initial, 'labels_lut':labels_lut}
    if not save_file:
        return samples

    nums = cu.match_filename(r'obd_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'obd_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_one_bead_sample(data_len, ks, diff_coefs, dt, n_data=1, n_steps_initial=10000,
    save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of one bead simulation trajectories for a given range of k and diff_coef values. as there is only one heat bath, this is a passive trajectory

    inputs:
    -------
    - data_len: int, length of each process
    - ks: numeric or length 2 list-like representing low, high values of the spring constant
    - diff_coefs: numeric or length 2 list-like representing low, high values of the diffusion coefficients
    - dt: time step between data points
    - n_data: number of processes in ensemble
    - n_steps_initial: number of steps to take in Langevin equation for simulating initial positions
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 1, data_len) which is an
        ensemble of one bead trajectories. the singleton dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are the ndarrays. 
            each ndarray is shaped (n_data,) whose values are the ks, diff_coefs values
        'label_names': list whose elements are string type, ['ks', 'diff_coefs']
        'dt': float type dt
        'n_steps_initial': int type n_steps_initial
    - (file_name): string type file name of the simulated data. returned if save_file is True

    FIXME: check the code to see if actual simulation part is not mixed up with initial condition simulation
    '''
    if isinstance(ks, (int, float)):
        ks = np.array([ks, ks], dtype=dtype)
    if isinstance(diff_coefs, (int, float)):
        diff_coefs = np.array([diff_coefs, diff_coefs], dtype=dtype)
    assert(len(ks) == 2), "Invalid ks given: should be numeric or length 2 list-like format"
    assert(len(diff_coefs) == 2), "Invalid diff_coefs given: should be numeric or length 2 list-like format"
    k_low, k_high = ks
    diff_coef_low, diff_coef_high = diff_coefs
    k_samples = (k_high - k_low) * np.random.random(n_data,) + k_low 
    diff_coef_samples = (diff_coef_high - diff_coef_low) * np.random.random(n_data,) + diff_coef_low 
    k_diff_coef_samples = np.stack([k_samples, diff_coef_samples], axis=1)

    concat_list = []
    for k_sample, diff_coef_sample in k_diff_coef_samples:
        process = sim_one_bead(data_len, ks=k_sample, diff_coefs=diff_coef_sample, dt=dt, n_data=1, n_steps_initial=n_steps_initial, save_file=False, dtype=dtype)
        process = process['data']
        concat_list.append(process)
    processes = np.concatenate(concat_list, axis=0) # shaped (n_data, 1, data_len)

    samples = {'data':processes, 'labels':[k_samples, diff_coef_samples], 'label_names':['ks', 'diff_coefs'], 'dt':dt, 'n_steps_initial':n_steps_initial}
    if not save_file:
        return samples

    nums = cu.match_filename(r'obd_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'obd_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_two_beads(data_len, gammas, k_ratios, diff_coef_ratios, dt, n_data=1, n_steps_initial=10000,
    save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of two bead simulation trajectories.

    inputs:
    -------
    - data_len: int, length of each process
    - gammas: numeric or list-like, drag coefficient values
    - k_ratios: numeric or list-like, ratios of spring constants
    - diff_coef_ratios: numeric or list-like, ratios of diffusion coefficients
    - dt: float, time step between data points
    - n_data: int, number of processes in ensemble
    - n_steps_initial: number of steps to take in Langevin equation for simulating initial positions
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 2, data_len) which is an
        ensemble of two beads trajectories. the 2nd dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are indices for the label values
        'label_names': list whose elements are string type, ['gammas', 'k_ratios', 'diff_coef_ratios']
        'labels_lut': list where the values are the label values given the index values in 'labels'
        'dt': float type dt
        'n_steps_initial': int type n_steps_initial
    - (file_name): string type file name of the simulated data. data returned if save_file is True

    FIXME: check the code to see if the dimensions are not mixed up, check if actual simulation part is not mixed up with initial condition simulation
    '''
    if isinstance(gammas, (int, float)):
        gammas = [gammas]
    if isinstance(k_ratios, (int, float)):
        k_ratios = [k_ratios]
    if isinstance(diff_coef_ratios, (int, float)):
        diff_coef_ratios = [diff_coef_ratios]

    gammas = np.array(gammas, dtype=dtype)
    k_ratios = np.array(k_ratios, dtype=dtype)
    diff_coef_ratios = np.array(diff_coef_ratios, dtype=dtype)
    n_gammas = len(gammas)
    n_k_ratios = len(k_ratios)
    n_diff_coef_ratios = len(diff_coef_ratios)

    file_size_est = data_len * n_gammas * n_diff_coef_ratios * n_k_ratios * n_data * 2 * np.dtype(dtype).itemsize
    file_size_est_gb = file_size_est / 1.e9
    if file_size_est_gb > 2.:
        warnings.warn("Generating file with size roughly {:.2f} GB".format(file_size_est_gb), Category=BytesWarning)

    processes = np.empty((n_gammas, n_k_ratios, n_diff_coef_ratios, n_data, 2, data_len)).astype(dtype)

    for idx0, gamma in enumerate(gammas):
        for idx1, k in enumerate(k_ratios):
            for idx2, diff_coef in enumerate(diff_coef_ratios):
                force_matrix = np.array([[-(1+k),k],[k,-(1+k)]])
                diffusion_matrix = np.array([[diff_coef,0],[0,1]])
                prefactor1 = force_matrix * dt
                prefactor2 = np.sqrt(2 * diffusion_matrix * dt)
                rand_nums = np.random.normal(0, 1, [n_steps_initial, 2, n_data])
                x0 = np.zeros((2, n_data))
                for idx in range(n_steps_initial):
                    x0 = x0 + np.matmul(prefactor1,x0) + np.matmul(prefactor2,rand_nums[idx])
                processes[idx0, idx1, idx2, :, :, 0] = x0.T

        for idx1, k in enumerate(k_ratios):
            for idx2, diff_coef in enumerate(diff_coef_ratios):
                x = x0
                force_matrix = np.array([[-(1+k),k],[k,-(1+k)]])
                diffusion_matrix = np.array([[diff_coef,0],[0,1]])
                prefactor1 = force_matrix * dt
                prefactor2 = np.sqrt(2 * diffusion_matrix * dt)
                rand_nums = np.random.normal(0, 1, [data_len - 1, 2, n_data])
                for idx in range(data_len - 1):
                    x = x + np.matmul(prefactor1,x) + np.matmul(prefactor2,rand_nums[idx])
                    processes[idx0, idx1, idx2, :, :, idx + 1] = x.T

        processes[idx0] = processes[idx0] / gamma

    # reshape data
    n_data_total = n_gammas * n_k_ratios * n_diff_coef_ratios * n_data
    processes = np.reshape(processes, (n_data_total, 2, data_len)) # shaped (n_data_total, 2, data_len)

    # reshape labels
    labels = [gammas, k_ratios, diff_coef_ratios]
    labels = np.array(list(product(*labels)), dtype=dtype) # shaped (n_conditions, n_labels)
    labels_lut = [tuple(condition) for condition in labels]
    n_conditions = len(labels_lut)

    labels = np.arange(n_conditions) # shaped (n_conditions,)
    labels = np.repeat(labels, n_data) # shaped (n_conditions * n_samples_total,)                                                       

    samples = {'data':processes, 'labels':labels, 'label_names':['gammas', 'k_ratios', 'diff_coefs'], 'dt':dt, 'n_steps_initial':n_steps_initial, 'labels_lut':labels_lut}
    if not save_file:
        return samples

    nums = cu.match_filename(r'tbd_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'tbd_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name

def sim_two_beads_sample(data_len, gammas, k_ratios, diff_coef_ratios, dt, n_data=1, n_steps_initial=10000,
    save_file=False, root_dir=ROOT_DIR, dtype='float32'):
    '''
    returns ensemble of two bead simulation trajectories for a given range of spring constant and diffusion coefficient values.

    inputs:
    -------
    - data_len: int, length of each process
    - gammas: numeric or length 2 list-like representing low, high values of the drag coefficient values
    - k_ratios: numeric or length 2 list-like representing low, high values of the ratios of spring constants
    - diff_coef_ratios: numeric or length 2 list-like representing low, high values of the ratios of diffusion coefficients
    - dt: float, time step between data points
    - n_data: int, number of processes in ensemble
    - n_steps_initial: number of steps to take in Langevin equation for simulating initial positions
    - save_file: boolean, whether or not to save the file. If True, file name is returned.
        Otherwise, data is returned
    - root_dir: string, root directory to save file if save_file is True
    - dtype: 'float32' or 'float64', precision of output data

    outputs:
    --------
    - (processes): dict whose key-value pairs are the following:
        'data': ndarray shaped (n_data, 2, data_len) which is an
        ensemble of two beads trajectories. the 2nd dimension is for the number of channels.
        returned if save_file is False
        'labels': list whose values are the ndarrays. 
            each ndarray is shaped (n_data,) whose values are the gammas, k_ratios, diff_coef_ratios values
        'label_names': list whose elements are string type, ['gammas', 'k_ratios', 'diff_coef_ratios']
        'dt': float type dt
        'n_steps_initial': int type n_steps_initial
    - (file_name): string type file name of the simulated data. data returned if save_file is True

    FIXME: check the code to see if actual simulation part is not mixed up with initial condition simulation
    '''
    if isinstance(gammas, (int, float)):
        gammas = np.array([gammas, gammas], dtype=dtype)
    if isinstance(k_ratios, (int, float)):
        k_ratios = np.array([k_ratios, k_ratios], dtype=dtype)
    if isinstance(diff_coef_ratios, (int, float)):
        diff_coef_ratios = np.array([diff_coef_ratios, diff_coef_ratios], dtype=dtype)
    assert(len(gammas) == 2), "Invalid gammas given: should be numeric or length 2 list-like format"
    assert(len(k_ratios) == 2), "Invalid k_ratios given: should be numeric or length 2 list-like format"
    assert(len(diff_coef_ratios) == 2), "Invalid diff_coef_ratios given: should be numeric or length 2 list-like format"
    gamma_low, gamma_high = gammas
    k_ratio_low, k_ratio_high = k_ratios
    diff_coef_ratio_low, diff_coef_ratio_high = diff_coef_ratios
    gamma_samples = (gamma_high - gamma_low) * np.random.random(n_data,) + gamma_low 
    k_ratio_samples = (k_ratio_high - k_ratio_low) * np.random.random(n_data,) + k_ratio_low 
    diff_coef_ratio_samples = (diff_coef_ratio_high - diff_coef_ratio_low) * np.random.random(n_data,) + diff_coef_ratio_low 
    param_samples = np.stack([gamma_samples, k_ratio_samples, diff_coef_ratio_samples], axis=1)

    concat_list = []
    for gamma_sample, k_ratio_sample, diff_coef_ratio_sample in param_samples:
        process = sim_two_beads(data_len, gammas=gamma_sample, k_ratios=k_ratio_sample,
                diff_coef_ratios=diff_coef_ratio_sample, dt=dt, n_data=1,
                n_steps_initial=n_steps_initial, save_file=False, dtype=dtype)
        process = process['data']
        concat_list.append(process)
    processes = np.concatenate(concat_list, axis=0) # shaped (n_data, 2, data_len)

    samples = {'data':processes, 'labels':[gamma_samples, k_ratio_samples, diff_coef_ratio_samples],
            'label_names':['gammas', 'k_ratios', 'diff_coef_ratios'], 'dt':dt, 'n_steps_initial':n_steps_initial}
    if not save_file:
        return samples

    nums = cu.match_filename(r'tbd_([0-9]+).pt', root_dir=root_dir)
    nums = [int(num) for num in nums]
    idx = max(nums) + 1 if nums else 0

    file_name = 'tbd_{}.pt'.format(idx)
    file_path = os.path.join(root_dir, file_name)
    torch.save(samples, file_path)
    return file_name
