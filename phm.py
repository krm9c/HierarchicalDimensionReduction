import random
import os,sys
import numpy as np
import warnings
import tflearn
import math


path_phm08 = '../data/PHM08'
def data_PHM08(path_phm08):
	temp_data_class_1 = []
	temp_data_class_2 = []
	Train_temp = np.loadtxt(path_phm08+'/train.txt')
	for i in xrange(1,219):
		index = [j for j,v in enumerate(Train_temp[:,0]) if v == i ]
		temp = Train_temp[index,:]
		temp_data_class_1.extend( temp[  0: int(math.ceil( (5/float(100))*temp.shape[0])), 5:temp.shape[1] ])
		temp_data_class_2.extend( temp[  int(math.ceil( (5/float(100))*temp.shape[0]))+1:temp.shape[0], 5:temp.shape[1] ])

	temp_data =[]
	temp_data.extend(temp_data_class_1)
	temp_data.extend(temp_data_class_2)
	y_train = np.concatenate( (np.zeros( len(temp_data_class_1) )+1, np.zeros( len(temp_data_class_2) )+2) )
	X_train = np.array(temp_data)

	temp_data_class_1 = []
	temp_data_class_2 = []
	Train_temp = np.loadtxt(path_phm08+'/test.txt')
	for i in xrange(1,219):
		index = [j for j,v in enumerate(Train_temp[:,0]) if v == i ]
		temp = Train_temp[index,:]
		temp_data_class_1.extend( temp[  0: int(math.ceil( (5/float(100))*temp.shape[0])) , 5:temp.shape[1] ])
		temp_data_class_2.extend( temp[  int(math.ceil( (5/float(100))*temp.shape[0]))+1:temp.shape[0] , 5:temp.shape[1] ])
	temp_data =[]
	temp_data.extend(temp_data_class_1)




	temp_data.extend(temp_data_class_2)
	y_test = np.concatenate( (np.zeros( len(temp_data_class_1) )+1, np.zeros( len(temp_data_class_2) )+2) )
	X_test = np.array(temp_data)

	print X_train.shape, y_train.shape, X_test.shape, y_test.shape

	T = np.concatenate((X_train, X_test))
	Y = np.concatenate((y_train, y_test))
	P = list(np.random.permutation(T.shape[0]))
	y = Y[P]
	T = T[P,:]

	print T.shape, y.shape
	return T, y

T, y = data_PHM08(path_phm08)
