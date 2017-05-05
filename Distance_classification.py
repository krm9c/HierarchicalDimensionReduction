import os, sys
import numpy as np
from random import seed
from random import random
from random import randrange
from scipy.stats import moment, norm
from math import sqrt
import matplotlib.pyplot as plt

sys.path.append('../CommonLibrariesDissertation')
path_data = '../data/Rolling_Data/'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *

#########################################################################
## The final classification
# Function to calculate the moments
def moments(D, order):
    return moment(D, moment=order)

# Evaluate the density
def density(ele, D):
    gamma = float(moments(D,3))
    Gamma = float(moments(D,4))
    h2= float(moments(D,2)-1)
    h3= float(moments(D,3)-3*moments(D,1))
    h5= float(moments(D,5)-10*moments(D,3)+15*moments(D,1))
    B = len(D)
    return (norm.pdf(ele)*(1+( (gamma *h2)/float(6*sqrt(B)) )+\
        ( ((Gamma-3)*h3)/float(24*B))  +\
        ( (pow(gamma,2)*h2)/float(72*B*B))))

# Evaluate the density
def density(num, ele):
    D = load_distance(num)
    gamma = float(moments(D,3))
    Gamma = float(moments(D,4))
    h2= float(moments(D,2)-1)
    h3= float(moments(D,3)-3*moments(D,1))
    h5= float(moments(D,5)-10*moments(D,3)+15*moments(D,1))
    B = len(D)
    return (norm.pdf(ele)*(1+((gamma*h2)/float(6*sqrt(B)))+\
        (((Gamma-3)*h3)/float(24*B))  +\
        ((pow(gamma,2)*h2)/float(72*B*B))))

def load_distance(num):
    D1=\
    np.loadtxt(path_data+'R'+str((num+1))+ dataset_type +'.csv', delimiter = ',')
    D1.astype(float)
    return preprocessing.scale(D1)


# Bayesian Classification Functions
def fisher(DM, C):
    # First let us decide on the prior probabilities
    P=[]
    Prob_label = []
    from tqdm import tqdm
    for j in tqdm(xrange(DM.shape[0])):
        element = DM[j,:];
        pi = norm.pdf(element)
        sum_density = sum([pi[i]*density(i, element[i]) for i in xrange(C)])
        P = [(pi[i]*density(i,element[i])) / float(sum_density) for i in xrange(C)]
        Prob_label.append( np.argmax(np.array(P)));
    print max(Prob_label), min(Prob_label)
    return np.resize((np.array(Prob_label)+1),(-1,))
## Bootstrap sample for the data
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Method for distance calculation
def Classification_Distances(Reference , Test):
    scaler = preprocessing.StandardScaler().fit(Reference)
    Ref, Tree = initialize_calculation(T = None, Data = scaler.transform(Reference), gsize = 3,\
    par_train = 0)
    T, Tree = initialize_calculation(T = Tree, Data = scaler.transform(Test), gsize = 3,\
    par_train = 1)

    return(traditional_MTS(Ref, T, par=0))

########################################################################
def Classification_Faults(N, T, y_true, C):

    for total in range(0,1):
        D_M= np.zeros((T.shape[0],C))
        for i in xrange(C):
            from tqdm import tqdm
            for k in tqdm(xrange(1000)):
                test = np.array(subsample(T, ratio=1.0))
                D_M[k,i] = np.reshape(Classification_Distances(N[i], test), [-1]).mean()
            D_M[:,i] = preprocessing.scale(D_M[:,i])

        y_pred = fisher(D_M,C)

        from sklearn.metrics import accuracy_score
        print y_true.shape, y_pred.shape
        print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    return T, y_pred


# Start with classification samples
def classification_start(C, dataset_type, dataset_num):
    R = []
    for i in xrange(1,C+1):
        Temp = np.loadtxt(path_data+'C'+str(i)+dataset_type+'.csv', delimiter = ',')
        R.append(Temp)
    R = np.array(R)
    T1, y_true = DataImport(num=dataset_num, classes=C)
    print R.shape, np.array(T1).shape
    T, y_pred = Classification_Faults(R,T1,y_true, 4)
    return y_pred, y_true

dataset_type = 'rolling'
dataset_num = 12
Class = 4
y_pred, y_true = classification_start(Class, dataset_type, dataset_num)
