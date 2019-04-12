import numpy as np
import random

def poisson(lamb=5, datalength=2**10, dt = 0.01):
    increment = np.random.poisson(lamb*dt, size=datalength)
    x = np.empty([datalength])
    for i in range(datalength):
        x[i] = np.sum(increment[0:i])
    return x
