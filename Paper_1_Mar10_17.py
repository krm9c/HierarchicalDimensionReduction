"""
Testing For Paper-I
"""
# import all the required Libraries
import math
import numpy as np
import time, os, sys
from tqdm import tqdm
# We have t set path before this
sys.path.append('/Users/krishnanraghavan/Documents/Research/CommonLibrariesDissertation')
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *
## Extract samples from the data
def extract_samples(X, y, c, p):
    index_1 = [i for i,v in enumerate(y) if v == p]
    index_2 = [i for i,v in enumerate(y) if v == c]
    N = X[index_1, :];
    yN = y[index_1];
    T = X[index_2, :];
    yT = y[index_2];
    return N, T, yN, yT

## First Let us import our two data-sets
## 1 -- First up is the Rolling Element Bearing Data-set
X, y = DataImport(num=11, classes=4, file=0, sample_size = 1000, features = 100)
C  = 4
## 2 -- Next let us figure out parsing the sensorless dataset
# X, y = DataImport(num=3, classes=4, file=0, sample_size = 1000, features = 100)
# C = 11


T_iterations = 10
Samp_Size = 5000
# Class wise distance evaluation
from Library_Paper_one import traditional_MTS
print "Start the iterations"
# Loop to evaluate distance value for all the classes.
for p in tqdm(xrange(1,C+1)):
    N, T, yN, yT = extract_samples(X, y, 1, p)
    scaler = preprocessing.StandardScaler().fit(N)
    N = scaler.transform(N)
    Ref, Tree = initialize_calculation(T = None, Data = N, gsize = 3, \
    par_train = 0)
    for c in tqdm(xrange(1,C+1)):
        N, T, yN, yT = extract_samples(X, y, c, p)
        T = scaler.transform(T)
        Test, Tree = initialize_calculation(T = Tree, Data = T, gsize = 3,\
        par_train = 1)
        Data=[]
        for i in tqdm(xrange(T_iterations)):
            rand = [random.randint(0,Test.shape[0]-1) for i in \
            xrange(Samp_Size)]
            T = Test[rand,:]
            rand = [random.randint(0,Ref.shape[0]-1) for i in \
            xrange(Samp_Size)]
            Tmp = traditional_MTS(Ref[rand,:], T, par=0)
            Data.append(Tmp)
        np.savetxt(('/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling/'+'R'+str(p)+'T'+str(c)+'rolling.csv'),\
        np.reshape(np.array(Data), [-1,]), delimiter =',')
