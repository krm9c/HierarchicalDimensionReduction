"""
Testing For Paper-I
"""
# import all the required Libraries
import math
import numpy as np
import time, os, sys
from tqdm import tqdm

## We have the set path before this
sys.path.append('../CommonLibrariesDissertation')
path_store = '../FinalDistSamples/'
from random import randrange

## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *

## Extract samples from the data
def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:];
    return N

## Bootstrap sample for the data
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample


## Distance iteration to extract and save samples
def distance_Iteration():
    ## Class wise distance evaluation
    from Library_Paper_one import traditional_MTS

    # # Loop to evaluate distance value for all the classes.
    for p in tqdm(xrange(1,C+1)):
        N = extract_samples(X, y, p)
        np.savetxt(  ("../"+'C'+str(p)+dataset_type+'.csv'), np.array(N), delimiter =',')
        scaler = preprocessing.StandardScaler().fit(N)
        N_transform = scaler.transform(N)

        Ref, Tree = initialize_calculation(T = None, Data = N_transform, gsize = 3, par_train = 0)
        Data = []

        np.savetxt(  ("../"+'C_dimred'+str(p)+dataset_type+'.csv'), np.array(Ref), delimiter =',')

        for i in tqdm(xrange(T_iterations)):
            T = np.array(subsample(N_transform, ratio=0.06))
            T_dim, Tree = initialize_calculation(T = Tree, Data = T, gsize = 10, par_train = 1)

            Tmp = traditional_MTS(Ref, T_dim, par=0)
            Data.append(Tmp.mean())

        np.savetxt(  ("../"+'R'+str(p)+dataset_type+'.csv'), np.reshape(np.array(Data), [-1,]), delimiter =',')

## First Let us import our two data-sets=
X, y = DataImport(num=3)

print X.shape
dataset_type = 'sensorless'
C = 11
T_iterations = 10000
distance_Iteration()
