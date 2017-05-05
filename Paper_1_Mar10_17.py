"""
Testing For Paper-I
"""
# import all the required Libraries
import math
import numpy as np
import time, os, sys
from tqdm import tqdm
# We have the set path before this
sys.path.append('../CommonLibrariesDissertation')
path_store = '../FinalDistSamples/'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *
## Extract samples from the data
def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:];
    return N

#  Distance calculation for just plain testing
def test_distance_calculation():
    # Start afresh again
    # Let us now test how our distance behaves
    # Get the first sample
    N = extract_samples(X, y, 1)
    Samp_Size = 10000
    rand = [numpy.random.choice(N.shape[0], Samp_Size, replace = True) for i in \
    xrange(Samp_Size)]
    N = N[rand,:]
    scaler = preprocessing.StandardScaler().fit(N)
    N_transform = scaler.transform(N)
    Ref, Tree = initialize_calculation(T = None, Data = N_transform, gsize = 3, \
    par_train = 0)

    # Get the other sample
    T = extract_samples(X, y, 4)
    rand = [numpy.random.choice(N.shape[0], Samp_Size, replace = True) for i in \
    xrange(Samp_Size)]
    T = T[rand,:]
    T_transform = scaler.transform(T)
    Test, Tree = initialize_calculation(T = Tree, Data = T_transform, gsize = 3,\
    par_train = 1)
    Tmp = traditional_MTS(Ref, Test, par=0)

    import matplotlib.pyplot as plt
    plt.plot(Tmp)
    plt.show()


# Generate samples for our testing and storage
def generate_samples(X, y, C, filename):
    Samp_Size = 10000
    for i in tqdm(xrange(1,C+1)):
        N = extract_samples(X, y, i)
        N = N[np.random.choice(N.shape[0],Samp_Size,replace= True), :]
        strfile = path_store+'C'+str(i)+filename+'.csv'
        np.savetxt(strfile,N, delimiter =',')

def distance_Iteration():
    ## Class wise distance evaluation
    from Library_Paper_one import traditional_MTS
    # # Loop to evaluate distance value for all the classes.
    for p in tqdm(xrange(1,C+1)):
        N = np.loadtxt(path_store+'C'+str(p)+dataset_type+'.csv', delimiter = ',')
        scaler = preprocessing.StandardScaler().fit(N)
        N_transform = scaler.transform(N)
        Ref, Tree = initialize_calculation(T = None, Data = N_transform, gsize = 3,\
         par_train = 0)

        T = np.loadtxt(path_store+'C'+str(p)+dataset_type+'.csv', delimiter=',')
        T_transform = scaler.transform(T)
        Test, Tree = initialize_calculation(T = Tree, Data = T_transform, gsize = 3,\
        par_train = 1)

        Data=[]
        for i in tqdm(xrange(T_iterations)):
            rand = np.random.choice(Test.shape[0],Samp_Size,replace= True)
            T = Test[rand,:]
            Tmp = traditional_MTS(Ref, T, par=0)
            Data.append(Tmp.mean())

        np.savetxt((path_store+'R'+str(p)+dataset_type+'.csv'), np.reshape(np.array(Data), [-1,]), delimiter =',')


## First Let us import our two data-sets
## 1 -- First up is the Rolling Element Bearing Data-set
# X, y = DataImport(num=0, classes = 20, sample_size = 10000)
## 2 -- Next let us figure out parsing the sensorless dataset
X, y = DataImport(num=12)

print "Shapes", X.shape, y.shape
C = 4
T_iterations = 10000
Samp_Size = 1000
dataset_type = 'rolling'
generate_samples(X, y, C, dataset_type)
# print "Samples are now ready, Iterate and evaluate distance "
distance_Iteration()
