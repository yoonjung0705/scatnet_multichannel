import numpy as np
import random

def brownian(diffusion=10, datalength=2**10, dt=0.01):
    x = np.empty([datalength])
    x[0] = random.gauss(0,1)
    for k in range(datalength-1):
        x[k+1] = x[k] + np.sqrt(2 * diffusion * dt)*random.gauss(0,1)

    return x
