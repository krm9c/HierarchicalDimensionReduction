####################################################################
# Header File Definitions
from math import *
import os,sys
import numpy as np
from sklearn import preprocessing
from numba import cuda, autojit
########################################################################

@autojit
def pairwise_distance(x):
    dx = x[..., np.newaxis] - x[np.newaxis, ...]
    d = np.array([dx,dx])
    D = (d**2).sum(axis=0)**0.5
    return D[:,:,0]

## Distance Correlation calculation
def distance_covariance(x,y):
    x = np.reshape(x, [-1,1])
    y = np.reshape(y, [-1,1])
    N = x.shape[0]

    # First distance Matrix
    A = pairwise_distance(x)
    one_n = np.ones((N,N))/N
    temp = one_n.dot(A)
    A = A - temp - A.dot(one_n) + temp.dot(one_n)
    
    # Second Distance Matrix
    B = pairwise_distance(x)
    temp = one_n.dot(B)
    B = B - temp - B.dot(one_n) + temp.dot(one_n)
        
    
    nu_xy = (1/float(N))*np.sqrt(np.sum(np.multiply(A, B)))
    nu_xx = (1/float(N))*np.sqrt(np.sum(A**2))
    nu_yy = (1/float(N))*np.sqrt(np.sum((B**2)))
    if nu_xx*nu_xy < 1e-3:
        return 1e-3
    else:
        return nu_xy/np.sqrt(nu_xx*nu_yy)


## Dependence Matrix calculation
def dependence_calculation(X):
    m  = X.shape[1];
    C = np.zeros((m,m))
    rng = np.random.RandomState(0)
    P = X[rng.randint(X.shape[0], size=500), :]
    p = np.arange(m)
    for i in p:
        for j in xrange(i+1):
            if i == j:
                C[i][j] = 1.0
            C[i][j] = distance_covariance(P[:,i],  P[:,j]);
            C[j][i] = C[i][j]
    return C

