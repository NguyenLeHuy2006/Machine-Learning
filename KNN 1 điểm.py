from __future__ import print_function
import numpy as np
from time import time 


d, N = 1000, 10000 
X = np.random.randn(N, d) 
z = np.random.randn(d)    


def dist_pp(z, x):
    d = z - x.reshape(z.shape) 
    return np.sum(d*d)

def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])
    return res

def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1) 
    z2 = np.sum(z*z)    
    return X2 + z2 - 2*X.dot(z) 

t1 = time()
D1 = dist_ps_naive(z, X)
print('naive point2set, running time:', time() - t1, 's')
t1 = time()
D2 = dist_ps_fast(z, X)
print('fast point2set , running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D1 - D2))