# lets start with some imports
import random
import os, sys
from math import *
import numpy as np
import warnings
from sklearn import preprocessing

# Localized definition imports
from DistanceCov import *
import SparsePCA

## lets do the dimensions reduction in loops
class level():
    def __init__(self):
        self.level_shuffling=[]
        self.group_transformation = []
        self.scaler= []
        self.G_LIST=[]
        self.flag = 0

# The new group reduction methodology
def novel_groups(T, g_size):
    R_label = [i for i in xrange(T.shape[1])]
    T_label = []
    start = 0;
    step = int(len(R_label)/float(g_size));
    for start in xrange(len(R_label)):
        T_label.extend( [ R_label[i] for i in xrange(start,int(len(R_label)),step)] )
        if(len(T_label)>=len(R_label)):
            break
    return np.array(T[:, T_label]), T_label

# Extract samples
def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:]
    return N

# Data In
def Low_Rank_Approx(r, Sigma):
    U, V= SparsePCA.sparse_pca(Sigma, r , 0.0001, maxit=10,\
    method='lars', n_jobs=1, verbose=0)
    return (U.reshape(Sigma.shape[1],r))

# Generate reduction vectors
def gen_red_vectors(r, Sigma):
    U, s, V = np.linalg.svd(Sigma)
    pc=  V[0:r,:]
    return pc.T


def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)
    return X_white

def dim_reduction(X, i_dim, o_dim, g_size, flag):
    Level =[]
    i_len = X.shape[0]
    Temp_X = X.copy()

    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size))< o_dim:
        Level.append(level())
        Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(with_mean = False, with_std = True).fit(X))
        D_scaled = Level[len(Level)-1].scaler[0].transform(X)
        if   flag =='corr':
            Sigma = np.nan_to_num(np.corrcoef(D_scaled.T)+0.00001*np.eye(D_scaled.shape[1]))
        elif flag == 'distcorr':
            Sigma = dependence_calculation(D_scaled)     +0.00001*np.eye(D_scaled.shape[1])
        if   flag == 'distcorr':
            V = Low_Rank_Approx(o_dim, Sigma)
        elif flag == 'corr':
            V = gen_red_vectors(o_dim, Sigma)
        Level[len(Level)-1].group_transformation.append(V)
        T = D_scaled.dot(V)
        return Level, T




    ## if the group wise reduction is larger than the required number of dimensions
    prev = 0
    i_dim = X.shape[1]
    Total_transf = []
    while i_dim >= o_dim:
        # Stopping conditions
        if (i_dim/float(g_size)) < o_dim:
            Final = X[:,0:o_dim]
            break
        elif prev == i_dim and Level[len(Level)-1].flag == 1:
            Final = X[:,0:prev]
            break
        
        # Initilize for the first level
        Level.append(level())
        if prev == i_dim:
            Level[len(Level)-1].flag = 1
        prev = i_dim

        # First create all the groups
        for i in xrange(0, i_dim, g_size):
            if (i+g_size) < i_dim and (i+2*g_size) > i_dim:
                F = i_dim
            else:
                F = i+g_size
            if F <= i_dim:
                Level[len(Level)-1].G_LIST.append([j for j in xrange(i,F)])
        if len(Level[len(Level)-1].G_LIST) == 0:
            break

        Transform = np.zeros(( X.shape[1] , len(Level[len(Level)-1].G_LIST) ))
        g_index_x = 0
        g_index_y = 0
        eigen_final = []

        for element in Level[len(Level)-1].G_LIST:
            temp = np.array(X[:, np.array(element)]).astype(float)
            Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(\
            with_mean = False, with_std = True).fit(temp))
            D_scaled = Level[len(Level)-1].scaler[len(Level[len(Level)-1].scaler)\
            -1].transform(temp)
            r = 1

            if flag =='corr':
                Sigma = np.nan_to_num(np.corrcoef(D_scaled.T))  +0.00001 * np.eye(D_scaled.shape[1])
            elif flag == 'distcorr':
                Sigma = dependence_calculation(D_scaled)   +0.00001*np.eye(D_scaled.shape[1])

            # The transformation vectors
            if flag =='corr':
                V = gen_red_vectors(r, Sigma)
            elif flag == 'distcorr':
                V = Low_Rank_Approx(r, Sigma)
                
            Transform[g_index_x:g_index_x+g_size, g_index_y] =  V[0:g_size, 0]
            g_index_x = g_index_x+g_size
            g_index_y = g_index_y+1
            Level[len(Level)-1].group_transformation.append(V)
            
        X  = X.dot(Transform)
        Total_transf.append(Transform)
        i_dim = X.shape[1]

    for i in xrange(len(Total_transf)):
        if i == 0:
            Final_Trans = Total_transf[i]
        else:
            Final_Trans = (Final_Trans).dot(Total_transf[i])

    Final_Trans = Final_Trans/float(np.linalg.norm(Final_Trans))
    Final = Temp_X.dot(Final_Trans)
    Final = Final[:, 0:o_dim]
    return Level, Final



def dim_reduction_test(X, Level, i_dim, o_dim, g_size):
    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size) < o_dim):
        print("Too many components required, Proposed method cannot be used, doing PCA instead")
        D_scaled = Level[len(Level)-1].scaler[0].transform(X)
        # Transform the required arrays
        T = D_scaled.dot(Level[len(Level)-1].group_transformation[0])
        return T
    p = 0
    while p <len(Level):
        Temp_proj =np.zeros([X.shape[0],1])
        for j, element in enumerate(Level[p].G_LIST):       
            temp = np.array(X[:, np.array(element)]).astype(float)
            # print "Before", temp.shape
            # temp = temp[:, (temp != 0).sum(axis=0) > 0]
            # print "After", temp.shape
            D_scaled = Level[p].scaler[j].transform(temp)
            Temp_proj = np.column_stack([Temp_proj, \
            D_scaled.dot(Level[p].group_transformation[j])])
        T = Temp_proj[:,1:Temp_proj.shape[1]]
        i_dim = X.shape[1]
        p = p+1

    if Level[len(Level)-1].flag is not 1:
        return X[:,0:o_dim]
    elif Level[len(Level)-1].flag is 1:
        return X