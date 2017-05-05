import os, sys
import numpy as np
from random import seed
from random import random
from random import randrange
from scipy.stats import moment, norm
from math import sqrt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

sys.path.append('/Users/krishnanraghavan/Documents/Research/CommonLibrariesDissertation')
path = '/Users/krishnanraghavan/Desktop/Simulations_Paper_1/Diagnostics/Rolling'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *
# sample extraction routine
def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N  = X[index_1, :];
    return N
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
        ( (pow(gamma,2)*h2)/float(72*B*B))\
        ))
# Evaluate the density
def density(num, ele):
    D = load_distance(num)
    gamma = float(moments(D,3))
    Gamma = float(moments(D,4))
    h2= float(moments(D,2)-1)
    h3= float(moments(D,3)-3*moments(D,1))
    h5= float(moments(D,5)-10*moments(D,3)+15*moments(D,1))
    B = len(D)
    return (norm.pdf(ele)*(1+( (gamma *h2)/float(6*sqrt(B)) )+\
        ( ((Gamma-3)*h3)/float(24*B))  +\
        ( (pow(gamma,2)*h2)/float(72*B*B))\
        ))
    # return norm.pdf(ele)
# Load the bootstrap samples
def load_distance(num):
    D1=\
    np.loadtxt('/Users/krishnanraghavan/Documents/Research/HierarchicalDimensionReduction/FinalDistSamples/'+'R'+str((num+1))+ dataset_type +'.csv', delimiter = ',')
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
# Method for distance calculation
def Classification_Distances(Reference , Test):
    scaler = preprocessing.StandardScaler().fit(Reference)
    Ref, Tree = initialize_calculation(T = None, Data = scaler.transform(Reference), gsize = 3, \
    par_train = 0)
    T, Tree = initialize_calculation(T = Tree, Data = scaler.transform(Test), gsize = 3,\
    par_train = 1)
    return(traditional_MTS(Ref, T, par=0))
# Method for classification
def Classification_Faults(N, T, y_true, C):
    for total in range(0,1):
        D_M= np.zeros((T.shape[0],C))
        for i in xrange(C):
            D_M[:,i]=np.reshape(Classification_Distances( N[i], T),[-1])
        preprocessing.scale(D_M)
        y_pred = fisher(D_M,C)
        from sklearn.metrics import accuracy_score
        print (accuracy_score(y_true[0:len(y_true)-1], y_pred, normalize=True, sample_weight=None))
    return T, y_pred

# Classification of the faults start here, so import data and everything
def classification_start(C, dataset_type, dataset_num):amples/'+'C'+str(i)+dataset_type+'.csv', delimiter = ',')
        R.append(Temp)
    R = np.asarray(R, dtype = float)
    T1, y_true = DataImport(num=dataset_num, classes=C)
    print R.shape, np.array(T1).shape
    T, y_pred = Classification_Faults(R,T1,y_true, 4)
    return T, (y_pred-1), T1, y_true

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X[0:len(Y),:], Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    plt.subplot(1, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    two_class = np.where(Y[:, 2])
    three_class = np.where(Y[:, 3])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray')

    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
               facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
               facecolors='none', linewidths=2, label='Class 2')
    plt.scatter(X[two_class, 0], X[two_class, 1], s=80, edgecolors='r',
               facecolors='none', linewidths=2, label='Class 3')
    plt.scatter(X[three_class, 0], X[three_class, 1], s=80, edgecolors='g',
                facecolors='none', linewidths=2, label='Class 4')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")

dataset_type = 'artificial'
dataset_num = 3
Class = 11
X, Y, T, y_true = classification_start(Class, dataset_type, dataset_num)
import tflearn
Y = tflearn.data_utils.to_categorical(Y, 12)
