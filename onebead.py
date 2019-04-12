import numpy as np
import random

def onebead(diffusion=10, datalength=2**10, k=1, dt=0.01):
    x = np.empty(datalength) 
    
    y = 0.0
    for i in range(10000):
        y = y - k*y*dt + np.sqrt(2 * diffusion * dt)*random.gauss(0,1)

    x[0] = y
    for i in range(datalength-1):
        x[i+1] = x[i] - k*x[i]*dt + np.sqrt(2 * diffusion * dt)*random.gauss(0,1)

    return x
