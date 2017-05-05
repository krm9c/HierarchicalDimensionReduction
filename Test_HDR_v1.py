import numpy as np
import gzip, cPickle



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



dataset = 'rolling'
def import_pickled_data(string):
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

import os,sys
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *


X_train, y_train, X_test, y_test = import_pickled_data(dataset)
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
print "Train, Test", X_train.shape, X_test.shape
# Next let us initialize the dimension reduction preprocessing
from sklearn.decomposition import PCA

# pca = PCA(n_components=12, svd_solver='full')
# Ref = pca.fit_transform(X_train )
# Test = pca.transform(X_test)

Ref, Tree = initialize_calculation(T = None, Data = X_train, gsize = 2,\
par_train = 0, output_dimension = 6)
Test, Tree = initialize_calculation(T = Tree, Data = X_test, gsize = 2,\
par_train = 1, output_dimension = 6)
print "Train, Test", Ref.shape, Test.shape


# Solvers
# neigh = KNeighborsClassifier(n_neighbors=100)
# neigh = GaussianNB()
# neigh = QuadraticDiscriminantAnalysis()
neigh= SVC(kernel="linear", C=0.025)

neigh.fit(Ref, y_train)
y_pred = neigh.predict(Test)
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_pred)
