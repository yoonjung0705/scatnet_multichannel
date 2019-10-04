import numpy as np
import random
import torch

ROOT_DIR = './data/'

def sim_brownian(data_len, diff_coefs, dt, n_data=1, file_name=None, root_dir=ROOT_DIR):
    '''
    returns ensemble of brownian trajectories

    inputs:
    -------
    - data_len: int, length of each process
    - diff_coef: numeric or list or ndarray, diffusion coefficient. 
    - dt: float, time step between data points
    - n_data: int, number of processes in ensemble

    outputs:
    --------
    - processes: ndarray shaped (n_diff_coefs, n_data, data_len) which is an ensemble of brownian trajectories

    FIXME: check if dimensionsare not mixed up
    '''
    concat_list = []
    for diff_coef in diff_coefs:
        increments = np.sqrt(2 * diff_coef * dt) * np.random.normal(0, 1, [n_data, data_len - 1])
        x0 = np.random.normal(0, 1, [n_data, 1])
        increments = np.concatenate([x0, increments], axis=1)
        processes = increments.cumsum(axis=1)
        concat_list.append(processes)

    processes = np.stack(concat_list, axis=0)

    if file_name is None:
        return processes
    else:
        file_path = os.path.join(root_dir, file_name)
        torch.save(processes, file_path)

def sim_one_bead(data_len, diff_coefs, ks, dt, n_data=1, n_steps_initial=10000, file_name=None, root_dir=ROOT_DIR):
    '''
    returns ensemble of one bead simulation trajectories. as there is only one heat bath, this is a passive trajectory

    inputs:
    -------
    - data_len: int, length of each process
    - diff_coef: numeric or list or ndarray, diffusion coefficient
    - k: numeric or list or ndarray, spring constant
    - dt: time step between data points
    - n_data: number of processes in ensemble
    - n_steps_initial: number of steps to take in Langevin equation for simulating initial positions

    outputs:
    --------
    - processes: ndarray shaped (n_ks, n_diff_coefs, n_data, data_len) which is an ensemble of one bead simulation trajectories

    FIXME: check the code to see if the dimensions are not mixed up, check if actual simulation part is not mixed up with initial condition simulation
    '''
    if isinstance(diff_coefs, (int, float)):
        diff_coefs = [diff_coefs]

    if isinstance(ks, (int, float)):
        ks = [ks]

    n_diff_coefs = len(diff_coefs)
    n_ks = len(ks)

    processes = np.empty((n_ks, n_diff_coefs, n_data, data_len))

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

    if file_name is None:
        return processes
    else:
        file_path = os.path.join(root_dir, file_name)
        torch.save(processes, file_path)

def sim_poisson(data_len, lams, dt, n_data=1, file_name=None, root_dir=ROOT_DIR):
    '''
    returns ensemble of poisson processes

    inputs:
    -------
    - data_len: int, length of each process
    - lams: numeric or list or ndarray, expectation per interval
    - dt: time step between data points
    - n_data: number of processes in ensemble

    outputs:
    --------
    - processes: ndarray shaped (n_lams, n_data, data_len) which is an ensemble of poisson processes

    REVIEW: confirm this method of using fixed time step generates identical statistics to that of Gielespie algorithm
    FIXME: confirm dimensions are not mixed up
    '''
    concat_list = []
    for lam in lams:
        increments = np.random.poisson(lam * dt, size=[n_data, data_len])
        processes = increments.cumsum(axis=1)
        concat_list.append(processes)
    processes = np.stack(concat_list, axis=0)
    
    if file_name is None:
        return processes
    else:
        file_path = os.path.join(root_dir, file_name)
        torch.save(processes, file_path)

def sim_two_beads(data_len, diff_coef_ratios, k_ratios, dt, n_data=1, n_steps_initial=10000, file_name=None, root_dir=ROOT_DIR):

    if isinstance(diff_coef_ratios, (int, float)):
        diff_coef_ratios = [diff_coef_ratios]

    if isinstance(k_ratios, (int, float)):
        k_ratios = [k_ratios]

    n_diff_coef_ratios = len(diff_coef_ratios)
    n_k_ratios = len(k_ratios)

    processes = np.empty((n_k_ratios, n_diff_coef_ratios, n_data, 2, data_len))

    for idx0, k in enumerate(k_ratios):
        for idx1, diff_coef in enumerate(diff_coef_ratios):
            force_matrix = np.array([[-(1+k),k],[k,-(1+k)]])
            diffusion_matrix = np.array([[diff_coef,0],[0,1]])
            prefactor1 = force_matrix * dt
            prefactor2 = np.sqrt(2 * diffusion_matrix * dt)
            rand_nums = np.random.normal(0, 1, [n_steps_initial, 2, n_data])
            x0 = np.zeros((2, n_data))
            for idx in range(n_steps_initial):
                x0 = x0 + np.matmul(prefactor1,x0) + np.matmul(prefactor2,rand_nums[idx])
            processes[idx0, idx1, :, :, 0] = x0.T

    for idx0, k in enumerate(k_ratios):
        for idx1, diff_coef in enumerate(diff_coef_ratios):
            x = x0
            force_matrix = np.array([[-(1+k),k],[k,-(1+k)]])
            diffusion_matrix = np.array([[diff_coef,0],[0,1]])
            prefactor1 = force_matrix * dt
            prefactor2 = np.sqrt(2 * diffusion_matrix * dt)
            rand_nums = np.random.normal(0, 1, [data_len - 1, 2, n_data])
            for idx in range(data_len - 1):
                x = x + np.matmul(prefactor1,x) + np.matmul(prefactor2,rand_nums[idx])
                processes[idx0, idx1, :, :, idx + 1] = x.T

    if file_name is None:
        return processes
    else:
        file_path = os.path.join(root_dir, file_name)
        torch.save(processes, file_path)
