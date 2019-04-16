import numpy as np
import random

def sim_brownian(data_len, diff_coef, dt, n_data=1):
    '''
    returns ensemble of brownian trajectories

    inputs:
    -------
    - data_len: length of each process
    - diff_coef: diffusion coefficient
    - dt: time step between data points
    - n_data: number of processes in ensemble

    outputs:
    --------
    - processes: ndarray shaped (n_data, data_len) which is an ensemble of brownian trajectories
    '''
    increments = np.sqrt(2 * diff_coef * dt) * np.random.normal(0, 1, [n_data, data_len - 1])
    x0 = np.random.normal(0, 1, [n_data, 1])
    increments = np.concatenate([x0, increments], axis=1)
    processes = increments.cumsum(axis=1)

    return processes

def sim_one_bead(data_len, diff_coef, k, dt, n_data=1, n_steps_initial=10000):
    '''
    returns ensemble of one bead simulation trajectories. as there is only one heat bath, this is a passive trajectory

    inputs:
    -------
    - data_len: length of each process
    - diff_coef: diffusion coefficient
    - k: spring constant
    - dt: time step between data points
    - n_data: number of processes in ensemble
    - n_steps_initial: number of steps to take in Langevin equation for simulating initial positions

    outputs:
    --------
    - processes: ndarray shaped (n_data, data_len) which is an ensemble of one bead simulation trajectories

    NOTE: the spatial complexity is max(n_data * n_steps_initial, n_data * data_len)
    unless n_steps_initial is extremely large the spatial complexity will be reasonably low and therefore we can cache the 
    random numbers for generating the initial conditions
    '''
    processes = np.empty([n_data, data_len])

    x0 = np.zeros(n_data)
    prefactor = np.sqrt(2 * diff_coef * dt)
    rand_nums = np.random.normal(0, 1, [n_data, n_steps_initial])
    for idx in range(n_steps_initial):
        x0 = x0 - k * dt * x0 + prefactor * rand_nums[:, idx]

    processes[:, 0] = x0
    rand_nums = np.random.normal(0, 1, [n_data, data_len - 1])
    for idx in range(data_len - 1):
        processes[:, idx + 1] = processes[:, idx] - k * dt * processes[:, idx] + prefactor * rand_nums[:, idx]

    return processes

def sim_poisson(data_len, lam, dt, n_data=1):
    '''
    returns ensemble of poisson processes

    inputs:
    -------
    - data_len: length of each process
    - lam: expectation per interval
    - dt: time step between data points
    - n_data: number of processes in ensemble

    outputs:
    --------
    - processes: ndarray shaped (n_data, data_len) which is an ensemble of poisson processes

    REVIEW: confirm this method of using fixed time step generates identical statistics to that of Gielespie algorithm
    '''
    increments = np.random.poisson(lam * dt, size=[n_data, data_len])
    processes = increments.cumsum(axis=1)
    
    return processes
