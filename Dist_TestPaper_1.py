import os
import numpy as np
from random import seed
from random import random
from random import randrange
from scipy.stats import moment, norm
from math import sqrt

path = '/Users/krishnanraghavan/Documents/Research/HierarchicalDimensionReduction/FinalDistSamples'

# Bootstrap sample from the data
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Functuion to calculate the probability for the fault 'R'
def data_pull(str):
	S=[]
	Samp_Size = 2500
	for file in os.listdir(path):
		if file.startswith(str):
			T = np.loadtxt(path+'/'+file);
			S.append(T[np.random.choice(T.shape[0],Samp_Size,replace= True)]);
	return np.reshape(np.array(S), [-1,])
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
C= 5
for p in tqdm(xrange(1,C)):
    D = data_pull('R'+str(p))
    np.savetxt(('C'+str(p)+'DistSample.csv'),\
    np.reshape(np.array(D), [-1,]), delimiter =',')
