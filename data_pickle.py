import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
import os, sys

from sklearn.model_selection import train_test_split

sys.path.append('../CommonLibrariesDissertation')
path_here = '../data/'

# We have to now import our libraries
from Data_import import *


def dump_data():
    p = [4]
    string = ['phm08']
    for i,s in enumerate(string):
        print s
        X, y = DataImport(num= p[i])
        X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.33, random_state=42)
        f = gzip.open(path_here+string[i]+'.pkl.gz','wb')
        dataset = [X_train, X_test, y_train, y_test]
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


def load_data():
    p = [0, 11, 2, 3, 4]
    string = ['phm08']
    for i,s in enumerate(string):
        f = gzip.open(path_here+string[i]+'.pkl.gz','rb')
        dataset = cPickle.load(f)
        X_train = dataset[0]
        X_test  = dataset[1]
        y_train = dataset[2]
        y_test  = dataset[3]
        print X_train.shape, y_train.shape, X_test.shape, y_test.shape




dump_data()
load_data()
