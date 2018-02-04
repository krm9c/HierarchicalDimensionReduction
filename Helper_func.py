# lets start with some imports
import random
import os, sys
from math import *
import numpy as np
from library_HDR_v1 import *
from random import randrange

#1 -- Helper function for data 
# Now lets setup the data
def import_pickled_data(string):
    import gzip, cPickle
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

# To collect samples
def extract_samples(X_f, y_f, p):
    index_1= [i for i,v in enumerate(y_f) if v == p]
    N = X_f[index_1,:]
    return N

# Bootstrap sample from the data
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
        print("dataset length", len(dataset))
        sample.append(dataset[index])
        print "sample goes", sample
	return sample

### 2 - Next the classification helper function
from sklearn.model_selection import train_test_split
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


# Gather data and setup HDR 
def classification_comparison(dataset, params, N):
    o_dim = params[1]
    g_size = params[0]
    flag = params[2]
    X,y,T,yT = import_pickled_data(dataset)
    # print(dataset, "is", X.shape)
    # #    Lets see how the dimension reduction
    P = np.zeros((N,9))
    score = np.zeros((N,9))
    for i in tqdm(xrange(N)):
        # DR -- 1 
        # Transform the training set
        # # Lets see how the dimension reduction
        Level, X_red_train = dim_reduction(X, i_dim=X.shape[1], o_dim =o_dim, \
        g_size=g_size, flag=flag)
        X_red_test=dim_reduction_test(T, Level, i_dim=X.shape[1], o_dim=o_dim,\
        g_size=g_size)
        # print("Dimensions reduced", X_red_train.shape, X_red_test.shape)
        # compare the classifiers
        names, score[i,:],  P[i,:] = comparison(X_red_train, y, X_red_test, yT)
    print("names", names)
    mean = np.round(score.mean(axis = 0),5)
    std  = np.round(score.std(axis = 0),5)
    s = ' '
    for i, element in enumerate(mean):
        s = s + ",("+str(element)+','+ str(std[i])+')' 

    print("Accuracy", s)
    mean = np.round(P.mean(axis = 0),5)
    std  = np.round(P.std(axis = 0), 5)
    s = ' '
    for i, element in enumerate(mean):
        s = s + ",("+str(element)+','+ str(std[i])+')' 
    print("p-value", s)


def comparison(XTrain, yTrain, XTest, yTest):

    names = ["Nearest Neighbors", "Linear SVM", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LDA"]

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(), 
    LinearDiscriminantAnalysis()
    ]
    s =[]
    score = []
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(XTrain, yTrain)
        score.append(clf.score(XTest, yTest))
        labels = clf.predict(XTest)
        i = int(max(labels)-1)
        p_value =0
        index = [p for p,v in enumerate(yTest) if v == i]
        index = [ int(x) for x in index ]
        yTest= [ int(x) for x in yTest ]
        L = [v for p,v in enumerate(labels) if p not in index]
        p_value = ( (list(L).count(i)) )/float(len(labels));
        s.append(p_value)
    return names, np.array(score).reshape(1,9), np.array(s).reshape(1,9)


### 2 - Next the helper function for comparing dimension reduction
def dim_reduction_comparison(dataset, n_comp):
    from sklearn.decomposition import FactorAnalysis, PCA, KernelPCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding
    from sklearn import preprocessing
    from tqdm import tqdm
    
    N, y_train, T, y_test = import_pickled_data(dataset)
    name_1 = ["FA", "KPCA", "Isomap", "LLE", "PCA"]

    dims = \
    [
    FactorAnalysis(n_components= n_comp, tol=0.01, \
     copy=True, max_iter=1000, noise_variance_init=None,\
     svd_method='randomized', iterated_power=3, random_state=0),

    KernelPCA(n_components= n_comp, kernel='linear', gamma=None, degree=3, \
     coef0=1, kernel_params=None, alpha=1.0, \
     fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None,\
     remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1),

    Isomap(n_neighbors=5, n_components=n_comp, eigen_solver= 'auto', tol=0, max_iter=None,\
     path_method='auto', neighbors_algorithm='auto', n_jobs=1),
    
    LocallyLinearEmbedding(n_neighbors=5, n_components=n_comp, reg=0.001, eigen_solver='auto',\
    tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12,\
    neighbors_algorithm= 'auto', random_state=None, n_jobs=1),

    PCA(n_components=n_comp, copy=True, whiten=False, svd_solver='auto', \
    tol=0.0, iterated_power= 'auto', random_state=None)
    ]

    # Transform the train data-set
    scaler = preprocessing.StandardScaler(with_mean = True,\
     with_std = True).fit(N)
    X_train = scaler.transform(N)
    X_test = scaler.transform(T)
    epoch = 1

    for n, clf in zip(name_1, dims):
        scores = np.zeros((epoch,9))
        p_value = np.zeros((epoch,9))
        print("DR is", n)
        for i in tqdm(xrange(epoch)):
            Train = clf.fit_transform(X_train)
            Test =  clf.transform(X_test)
            names, scores[i,:], p_value[i,:] = comparison(Train, y_train, Test, y_test)
        
        print("names", names)
        mean = np.round(scores.mean(axis = 0),5)
        std  = np.round(scores.std(axis = 0),5)
        s = ' '
        for i, element in enumerate(mean):
            s = s + ",("+str(element)+','+ str(std[i])+')' 
        print("Accuracy", s)
        mean = np.round(p_value.mean(axis = 0),5)
        std  = np.round(p_value.std(axis = 0), 5)
        s = ' '
        for i, element in enumerate(mean):
            s = s + ",("+str(element)+','+ str(std[i])+')' 
        print("p-value", s)


# 3-- Distance Values

def traditional_MTS(Norm, T, par):

    # Define the sizes of the normal and the test data set
    row_normal    = Norm.shape[0]
    column_normal = Norm.shape[1]
    row_test      = T.shape[0]
    column_test   = T.shape[1]

    # use the mean of both to transform each of the array
    scalar        =  preprocessing.StandardScaler(with_mean = True, with_std = True).fit(Norm)
    N_scaled      =  scalar.transform(Norm)
    T_scaled      =  scalar.transform(T)

    # Since we have the parameters now today, lets start with the transformation..
    mean_test=[]
    for i in range(0, column_test):
        a= np.array(T_scaled[:, i])
        mean_test.append(a.mean())
    mean_normal= []
    for i in range(0, column_test):
        a= np.array(N_scaled[:, i])
        mean_normal.append(a.mean())
    # Perform the final transformation
    # 1. if less than penultimate level (dimension reduction)
    # 2. If at penultimate level (distance calculation)
    mn           = np.array(mean_normal)
    mt           = np.array(mean_test)
    MD_test = np.zeros((T_scaled.shape[0],1))
    Correlation_Matrix = np.corrcoef(N_scaled.T)
    Correlation_Matrix = Correlation_Matrix+0.01*np.eye(Correlation_Matrix.shape[1])
    Inverse_correlation=np.linalg.pinv(Correlation_Matrix)
    # print "The norm value", (np.dot((mt-mn),np.transpose((mt-mn))))
    if par ==0:
        # Calculate the value of the MD
        for i in range (0,row_test):
            if ((np.dot(np.dot((T_scaled[i,:]-mn),Inverse_correlation),(T_scaled[i,:]-mn).transpose())) < 0):
                print ("problem")
            MD_test[i] = np.linalg.norm(np.dot((T_scaled[i,:]-mn),np.linalg.cholesky(Inverse_correlation)), 2);
    else:
        # Calculate the value of the MD
        for i in range (0,row_test):
                MD_test[i]=np.linalg.norm(np.dot((mt-mn),np.linalg.cholesky(Inverse_correlation)),2);
    return MD_test

# 4 -- Reduction of dimension
def reduced_dimension_data(XTr, TTe, params):
    from library_HDR_v1 import dim_reduction, dim_reduction_test
    o_dim = params[1]
    g_size = params[0]
    flag = params[2]
    
    Level, X_red_train = dim_reduction(XTr, i_dim=XTr.shape[1], o_dim =o_dim, \
    g_size=g_size, flag=flag)
    
    X_red_test=dim_reduction_test(TTe, Level, i_dim=TTe.shape[1], o_dim=o_dim,\
    g_size=g_size)
    
    return X_red_train, X_red_test
