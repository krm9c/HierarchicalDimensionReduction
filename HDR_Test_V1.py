
# First of all the imports 
# lets start with some imports
import os, sys
import numpy as np
from tqdm import tqdm
import random 

from Helper_func import import_pickled_data, extract_samples, subsample, traditional_MTS
from random import randrange

def Boot_sample(normal, sam_size):
    B_sample  = np.zeros((sam_size, normal.shape[1]))
    for i in xrange(sam_size):
        rand = [ random.randint(0,(normal.shape[0]-1)) for j in xrange(1) ]
        sample = normal[rand, :]
        B_sample[i,:] = sample.mean(axis= 0)
    return B_sample

from library_HDR_v1 import *

def distance_calculation_power(Sample1, Sample2, o_dim, g_size):
    Level, X_red_train = dim_reduction(Sample1, i_dim=Sample1.shape[1], o_dim =o_dim, \
        g_size=g_size, flag = 'corr')
    X_red_test=dim_reduction_test(Sample2, Level, i_dim=Sample2.shape[1], o_dim=o_dim,\
        g_size=g_size)
    MD = traditional_MTS(X_red_train, X_red_test, 0)
    return MD


def p_val(N, data_name, g_size, dim):
    
    # Sensorless
    p_values = np.zeros((N,3))
    X,y,T,yT = import_pickled_data(data_name)


    for i in tqdm(xrange(N)):
        # Extract Sample, Lets get data 
        Sample_normal = extract_samples(X, y, 0)
        scalar        =  preprocessing.StandardScaler(with_mean = True, with_std = True).fit(Sample_normal)
        N_scaled      =  scalar.transform(Sample_normal)
        Boot_strap_normal = Boot_sample(N_scaled, 10000)


        Sample_T =  extract_samples(T, yT, 0)
        T_scaled  =  scalar.transform(Sample_T)

        ### Results
        Distances = distance_calculation_power(Boot_strap_normal, Boot_strap_normal, dim, g_size)
        c = [np.percentile(Distances, 99), np.percentile(Distances, 95), np.percentile(Distances, 90)]
        Distances = distance_calculation_power( Boot_strap_normal, T_scaled, dim, g_size )

        p_values[i,0] = len([1 for j in Distances if j  > c[0]])/float(10000)
        p_values[i,1] = len([1 for j in Distances if j  > c[1]])/float(10000) 
        p_values[i,2] = len([1 for j in Distances if j  > c[2]])/float(10000) 
    print("P value -- mean, std", np.mean(p_values, axis= 0), np.std(p_values, axis=0))
    return Sample_normal, Boot_strap_normal, c

def power(N, Boot_strap_normal, Samp_Norm, data_name, c, g_size, dim, C):
    power = np.zeros((N,3))
    # Extract Sample, Lets get data 
    X,y,T,yT = import_pickled_data(data_name)
    scalar  =  preprocessing.StandardScaler(with_mean = True, with_std = True).fit(Samp_Norm)
    for i in tqdm(xrange(N)):   
        Rand_class = random.randint(1,C)  
        Sample_test = extract_samples(T, yT, Rand_class)
        T_scaled = scalar.transform(Sample_test)

        ### Results
        Distances = distance_calculation_power(Boot_strap_normal, T_scaled,  dim, g_size)
        power[i,0] = len([1 for j in Distances if j  < c[0]])/float(10000)
        power[i,1] = len([1 for j in Distances if j  < c[1]])/float(10000) 
        power[i,2] = len([1 for j in Distances if j  < c[2]])/float(10000) 

    print("Power -- mean std", np.mean(power, axis= 0), np.std(power, axis=0))


# Testing for p_values and power of the bootstrap test 
N = 1000
print("rolling")
Samp_Norm, Boot_strap_normal, c = p_val(N, 'rolling', 2, 5)
power(N, Boot_strap_normal, Samp_Norm, 'rolling', c, 2, 5, 3)
print("sensorless")
Samp_Norm, Boot_strap_sensorless,c = p_val(N, 'sensorless',  2, 20)
power(N, Boot_strap_sensorless, Samp_Norm, 'sensorless', c, 2, 20, 9)

