import os, sys
import numpy as np
from random import seed
from random import random
from random import randrange
from scipy.stats import moment, norm
from math import sqrt
sys.path.append('/Users/krishnanraghavan/Documents/Research/CommonLibrariesDissertation')
path = '/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *

def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N  = X[index_1, :];
    return N

## The final classification
# Function to calculate the moments
def moments(D, order):
    return moment(D, moment=order)

# Evaluate the density
def density(num, ele):
    if (num+1) ==1:
        D = D1;
    elif (num+1) ==2:
        D = D2;
    elif (num+1) ==3:
        D = D3;
    elif (num+1) ==4:
        D = D4;
    gamma = float(moments(D,3))
    Gamma = float(moments(D,4))
    h2= float(moments(D,2)-1)
    h3= float(moments(D,3)-3*moments(D,1))
    h5= float(moments(D,5)-10*moments(D,3)+15*moments(D,1))
    B = len(D)
    return (norm.pdf(ele)* (1+( (gamma *h2)/float(6*sqrt(B)) )+\
        ( ((Gamma-3)*h3)/float(24*B))  +\
        ( (pow(gamma,2)*h2)/float(72*B*B))\
        ))
# Load the bootstrap samples
def load_distance():
    global D1
    D1=\
    np.loadtxt('/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling/'+'C'+str(1)+'DistSample.csv', delimiter = ',')
    D1 = preprocessing.scale(D1)
    global D2
    D2 =\
    np.loadtxt('/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling/'+'C'+str(2)+'DistSample.csv', delimiter = ',')
    D2 = preprocessing.scale(D2)
    global D3
    D3  =\
    np.loadtxt('/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling/'+'C'+str(3)+'DistSample.csv', delimiter = ',')
    D3 = preprocessing.scale(D3)
    global D4
    D4 =\
    np.loadtxt('/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling/'+'C'+str(4)+'DistSample.csv', delimiter = ',')
    D4 = preprocessing.scale(D4)
# Bayesian Classification Functions
def fisher(DM, C):
    load_distance();
    # First let us decide on the prior probabilities
    pi = [1/float(C) for i in xrange(C)]
    P=[]
    Prob_label = []
    from tqdm import tqdm
    for j in tqdm(xrange(DM.shape[0])):
        element = DM[j,:];
        sum_density = sum([ pi[i]*density(i,element[i]) for i in xrange(C) ])
        P = [((pi[i]*density(1,element[i]))/float(sum_density)) for i in xrange(C)]
        Prob_label.append(np.argmax(np.array(P)));
    return np.resize(np.array(Prob_label),(-1,))
# Method for distance calculation
def Classification_Distances( Norm , T):
    scaler = preprocessing.StandardScaler().fit(Norm)
    Norm = scaler.transform(Norm)
    Ref, Tree = initialize_calculation(T = None, Data = Norm, gsize = 3, \
    par_train = 0)
    T = scaler.transform(T)
    Test, Tree = initialize_calculation(T = Tree, Data = T, gsize = 3,\
    par_train = 1)
    return(traditional_MTS(Norm, T, par=0))
# Method for classification
def Classification_Faults(N, T, y_true, C):
    for total in range(0,1):
        D_M= np.zeros((T.shape[0],C))
        for i in xrange(C):
            D_M[:,i]=preprocessing.scale(np.reshape(Classification_Distances( N[i], T),[-1]))
        y_pred = fisher(D_M,C)
        from sklearn.metrics import accuracy_score
        print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
# Classification of the faults start here
def classification_start():
    X, y = DataImport(num=12, classes=4, file=0, sample_size = 5000, features = 100)
    C  = 4
    IR = extract_samples(X, y, 3)
    OR = extract_samples(X, y, 4)
    NL = extract_samples(X, y, 2)
    Norm= extract_samples(X, y,1)
    R= []
    R.append(Norm)
    R.append(NL)
    R.append(IR)
    R.append(OR)
    R = np.array(R)
    T, y = DataImport(num=11, classes=4, file=0, sample_size = 1000, features = 100)
    print R.shape, np.array(T).shape
    Classification_Faults(R,T,y, 4)
classification_start()
