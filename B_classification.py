import os, sys
import numpy as np
from random import seed
from random import random
from random import randrange
from scipy.stats import moment, norm
from math import sqrt
import matplotlib.pyplot as plt




# Classifiers
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




sys.path.append('../CommonLibrariesDissertation')
path_data = '../data/Sensorless_data/'
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
    pi = [1/float(C) for i in xrange(C)]
    from tqdm import tqdm
    for j in tqdm(xrange(DM.shape[0])):
        element = DM[j,:];
        sum_density = sum([pi[i]*density(i, element[i]) for i in xrange(C)])
        P = [(pi[i]*density(i,element[i])) / float(sum_density) for i in xrange(C)]
        Prob_label.append( np.argmax(np.array(P)));
    print max(Prob_label), min(Prob_label)
    return (np.array(Prob_label)+1)

# Method for distance calculation
def Classification_Distances(Reference , Test):
    scaler = preprocessing.StandardScaler().fit(Reference)
    Ref, Tree = initialize_calculation(T = None, Data = scaler.transform(Reference), gsize = 3,\
    par_train = 0)
    T, Tree = initialize_calculation(T = Tree, Data = scaler.transform(Test), gsize = 3,\
    par_train = 1)
    return(traditional_MTS(Ref, T, par=0))

def Classification(Reference , Test):
    scaler = preprocessing.StandardScaler().fit(Reference)
    Ref, Tree = initialize_calculation(T = None, Data = scaler.transform(Reference), gsize = 10,\
    par_train = 0)
    T, Tree = initialize_calculation(T = Tree, Data = scaler.transform(Test), gsize = 10,\
    par_train = 1)
    return T

########################################################################
def Classification_Faults(N, T, y_true, C):
    for total in range(0,1):
        D_M= np.zeros((T.shape[0],C))
        for i in xrange(C):
            D_M[:,i] = np.reshape(Classification_Distances(N[i], T), [-1])

        y_pred = fisher(D_M,C)
        print y_pred
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

    # from sklearn.naive_bayes import GaussianNB
    # T = Classification(T1, T1)
    # print "Sensorless, with HDR"
    # print "1 -- GaussianNB"
    # gnb = GaussianNB()
    # y_pred = gnb.fit(T, y_true).predict(T)
    # from sklearn.metrics import accuracy_score
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    #
    # print "2 -- Quadratic"
    # clf = QuadraticDiscriminantAnalysis()
    # y_pred = clf.fit(T, y_true).predict(T)
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    # print "3 -- SVC"
    # clf = SVC(kernel="linear", C=0.025)
    # y_pred = clf.fit(T, y_true).predict(T)
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    #
    # print "4 -- KNN"
    # clf = KNeighborsClassifier(3)
    # y_pred = clf.fit(T, y_true).predict(T)
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    #
    # print "Sensorless, without HDR"
    # T = T1
    # print "1 -- GaussianNB"
    # gnb = GaussianNB()
    # y_pred = gnb.fit(T, y_true).predict(T)
    # from sklearn.metrics import accuracy_score
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    # print "2 -- Quadratic"
    # clf = QuadraticDiscriminantAnalysis()
    # y_pred = clf.fit(T, y_true).predict(T)
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    # print "3 -- SVC"
    # clf = SVC(kernel="linear", C=0.025)
    # y_pred = clf.fit(T, y_true).predict(T)
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))
    #
    #
    # print "4 -- KNN"
    # clf = KNeighborsClassifier(3)
    # y_pred = clf.fit(T, y_true).predict(T)
    # print (accuracy_score(y_true, y_pred, normalize=True, sample_weight=None))



    T, y_pred = Classification_Faults(R,T1,y_true, 11)
    return y_pred, y_true

dataset_type = 'sensorless'
dataset_num = 3
Class = 11
y_pred, y_true = classification_start(Class, dataset_type, dataset_num)
