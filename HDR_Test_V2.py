
# First of all the imports 
# lets start with some imports
import os, sys
import numpy as np
from tqdm import tqdm
import random
from scipy.stats import moment, norm

from Helper_func import import_pickled_data, extract_samples, subsample, traditional_MTS
from random import randrange

from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

def Boot_sample(normal, sam_size):
    B_sample  = np.zeros((sam_size, normal.shape[1]))
    for i in xrange(sam_size):
        rand = [ random.randint(0,(normal.shape[0]-1)) for j in xrange(10) ]
        sample = normal[rand, :]
        B_sample[i,:] = sample.mean(axis= 0)
    return B_sample

from library_HDR_v1 import *
def distance_calculation(Sample1, o_dim, g_size):
    Level, X_red_train = dim_reduction(Sample1, i_dim=Sample1.shape[1], o_dim =o_dim, \
        g_size=g_size, flag = 'corr')
    MD = traditional_MTS(X_red_train, X_red_train, 1)
    return MD

def distance_calculation_power(Sample1, Sample2, o_dim, g_size):
    Level, X_red_train = dim_reduction(Sample1, i_dim=Sample1.shape[1], o_dim =o_dim, \
        g_size=g_size, flag = 'corr')
    X_red_test=dim_reduction_test(Sample2, Level, i_dim=Sample2.shape[1], o_dim=o_dim,\
        g_size=g_size)
    MD = traditional_MTS(X_red_train, X_red_test, 1)
    print(MD.shape, MD.mean())
    return MD


def Generate_samples_each_class(N, data_name, classes):
    # Sensorless
    Distance = np.zeros((N,3))
    X,y,T,yT = import_pickled_data(data_name)
    for i in tqdm(xrange(N)):
        for j in tqdm(xrange(classes)):
            # Extract Sample, Lets get data 
            Sample_normal = extract_samples(X, y, j)
            Boot_strap_normal = Boot_sample(Sample_normal, 10000)
            #####
            np.savetxt('Data_Boot'+str(j)+data_name+'.csv', Boot_strap_normal, delimiter = ',' )


### KDE based Naive bayes classifier
def separate_class_samples(Train, labels):
    classes = int(np.max(labels)+1)
    Samples = []
    for i in xrange(classes):
        Samples.append( extract_samples(Train, labels, i) )
    return Samples

## Classification Method
def moments(D, order):
    return np.array( [ moment(D[:, i], moment=order ) for i in xrange(D.shape[1])]  )

# Evaluate the density
def density(sample, ele, pdf):
    moment_3 = moments(sample,3)
    moment_4 = moments(sample,4)
    moment_1 = moments(sample,1)
    moment_2 = moments(sample,2)
    moment_5 = moments(sample,5)
    h1 = np.sum(moment_1)
    h2 = np.sum(moment_2-1)
    h3 = np.sum(moment_3-3*moment_1)
    h5 = np.sum(moment_5-10*moment_3+15*moment_1)
    B = sample.shape[0]
    # Check this out through math, edgeworth expansion for multivariate densities
    factor = np.sum( 1+((moment_3*h2)/float(6*sqrt(B)))+\
        (((moment_4-3)*h3)/float(24*B))  +\
        ((np.power(moment_3,2)*h2)/float(72*B*B)))
    return np.exp(pdf.score_samples(ele))*factor

import operator
def bayes_classifier(x_vec, kdes):
    """
    Classifies an input sample into class w_j determined by
    maximizing the class conditional probability for p(x|w_j).
    Keyword arguments:
        x_vec: A dx1 dimensional numpy array representing the sample.
        kdes: List of the gausssian_kde (kernel density) estimates
    Returns a tuple ( p(x|w_j)_value, class label ).
    """
    p_vals = []
    for kde in kdes:
        p_vals.append( np.exp(kde.score_samples(x_vec)) )
    P_vals = np.array(p_vals).T
    print P_vals[100,:]    
    predict_labels = [ np.argmax( P_vals[i, :]/float(np.sum(P_vals[i,:])+0.01) ) for i in xrange(P_vals.shape[0])]
    return predict_labels


from Helper_func import extract_samples
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.model_selection import GridSearchCV
def classify_samples(XTrain, yTrain):
    n_classes = int(max(yTrain)+1)
    pdf_d =[]
    for i in range(n_classes):
        # use grid search cross-validation to optimize the bandwidth
        temp = extract_samples(XTrain, yTrain, i)
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(temp)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        kde_skl = grid.best_estimator_
        pdf_d.append(kde_skl)
    return pdf_d

from Helper_func import reduced_dimension_data
## Lets do this for all data-sets and how this works
data_name = ['rolling','sensorless', 'notmnist', 'arcene', 'gisette','dexter' ]
param = [ [2,5,'corr'], [2, 24,'corr'], [2, 392,'corr'], [100, 100,'corr'], [10, 50,'corr'], [100, 20,'corr']  ]
for i,data in enumerate(data_name):
    from Helper_func import import_pickled_data
    XTr,yTr,TTe,yTe = import_pickled_data(data)
    print("data", data)
    XTr, XTe = reduced_dimension_data(XTr, TTe, param[i])
    ## Lets reduce the data first 
    print("Shapes", XTr.shape, XTe.shape, yTr.shape, yTe.shape)
    pdf = classify_samples(XTr, yTr)
    print(len(pdf))
    predict =  bayes_classifier(XTe, pdf)
    from sklearn.metrics import accuracy_score
    print accuracy_score(yTe, predict, normalize=True, sample_weight=None)

