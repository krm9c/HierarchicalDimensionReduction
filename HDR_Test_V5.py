
# First of all the imports 
# lets start with some imports

import os, sys
import numpy as np
import random
from Helper_func import import_pickled_data, extract_samples, subsample, traditional_MTS
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math

def gen_data(dim, N_size):
    from random import choice
    from sklearn.datasets import make_spd_matrix
    # Define the number of samples 
    num_samples = N_size
    C = make_spd_matrix(dim, random_state=123) + 0.000001*np.identity(dim)
    mean = [0 for i in xrange(dim)] 
    X = np.random.multivariate_normal(mean, C,num_samples)    
    return X, C


def Dimension_Reduction():
    from Helper_func import reduced_dimension_data  
    n_times = 1
    n_size = [4096,2048,1024,512, 256, 128, 56, 10, 5] 
    r2 =np.zeros( (n_times,len(n_size)) )
    dim_data = 50
    from tqdm import tqdm
    n_comp = [5, 10, 15, 20]
    for comp in n_comp:   
        for i in tqdm(xrange(n_times)):
                dims =PCA(n_components= comp)
                for j, n in enumerate(n_size):
                    N, C = gen_data(dim_data, N_size= n)
                        
                    # Transform the train data-set
                    scaler = preprocessing.StandardScaler(with_mean = True,\
                        with_std = True).fit(N)
                    X_train = scaler.transform(N)

                        # Reduced dimensions
                    Train_PCA = dims.fit_transform(X_train)
                    Train_HDR, Test_HDR = reduced_dimension_data(X_train, X_train, [2, comp, 'corr'])


                    from sklearn.linear_model import LinearRegression        
                    lm = LinearRegression()
                    lm.fit(Train_PCA, Train_HDR)
                    r2[i,j] = lm.score(Train_PCA, Train_HDR)
                    
        print("comp", comp, "mean r2 square", r2.mean(axis = 0))

from DistanceCov import dependence_calculation
import timeit
import time
start = time.time()
Dimension_Reduction()
print(time.time()-start )
