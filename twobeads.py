import numpy as np

def twobeads(T1=500, T2=50, datalength=2**10, k=1, dt=0.01):
    x = np.empty((datalength,2))
    A = np.array([[-2*k,k],[k,-2*k]])
    diffusion = np.array([[T1,0],[0,T2]])

    y = np.zeros((2,1))
    for i in range(10000):
        y = y + np.matmul(A,y)*dt + np.matmul(np.sqrt(2 * diffusion * dt),np.random.normal(0,1,[2,1]))

    x[:,[0]] = y
    for i in range(datalength-1):
        x[:,[i+1]] = x[:,[i]] + np.matmul(A,x[:,[i]])*dt + np.matmul(np.sqrt(2 * diffusion * dt),np.random.normal(0,1,[2,1]))

    return x
