import numpy as np
import random

def brownian(diff_coef=10, data_len=2**10, dt=0.01):
    x = np.empty([data_len])
    x[0] = random.gauss(0,1)
    for k in range(data_len-1):
        x[k+1] = x[k] + np.sqrt(2 * diff_coef * dt)*random.gauss(0,1)

    return x

def onebead(diff_coef=10, data_len=2**10, k=1, dt=0.01):
    x = np.empty(data_len) 
    
    y = 0.0
    for i in range(10000):
        y = y - k*y*dt + np.sqrt(2 * diff_coef * dt)*random.gauss(0,1)
    x[0] = y
    for i in range(data_len-1):
        x[i+1] = x[i] - k*x[i]*dt + np.sqrt(2 * diff_coef * dt)*random.gauss(0,1)
    return x

def poisson(data_len=2**10, lamb=5, dt=0.01):
    increment = np.random.poisson(lamb*dt, size=data_len)
    return increment.cumsum()
    #x = np.empty([data_len])
    #for i in range(data_len):
    #    x[i] = np.sum(increment[0:i])
    #return x
