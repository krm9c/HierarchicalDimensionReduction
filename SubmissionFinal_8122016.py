"""
Created on Fri Apr 22 12:36:08 2016
@author: Krishnan Raghavan
"""
######################################################################################
# Pre Existing Libraries
from math import *
import os,sys
import random
import time
import numpy as np
import itertools
import math
from collections import Counter
from numpy import genfromtxt
import heapq
from   sklearn.decomposition         import PCA
import matplotlib.pyplot                 as plt
from   matplotlib.colors             import ListedColormap
from   sklearn.cross_validation      import train_test_split
from   sklearn.preprocessing         import StandardScaler
from   sklearn.datasets              import make_moons, make_circles, make_classification
from   sklearn.neighbors             import KNeighborsClassifier
from   sklearn.svm                   import SVC
from   sklearn.tree                  import DecisionTreeClassifier
from   sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier
from   sklearn.naive_bayes           import GaussianNB
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.random_projection 		 import SparseRandomProjection
from sklearn 						 import manifold
from   sklearn                       import cluster, datasets
from   sklearn.neighbors             import kneighbors_graph
from   sklearn.preprocessing         import StandardScaler

#######################################################################################
# Libraries created by us
sys.path.append('C:\Users\krm9c\Dropbox\Work\Research\Common_Libraries')
path= "E:\Research_Krishnan\Data\Data-case-study-1"
from Library_Paper_two       import *
from Library_Paper_one       import import_data, traditional_MTS


#######################################################################
# Distance Calculation for classification
def Classification_Distances( Norm, T, par, numberFlag, dimension):
	# print("(row,column)", "(",row_normal, column_normal,")");
	# print("(row,column)", "(",row_test, column_test,")");
	CurrentFile = [ ];

	# MD = traditional_MTS(Norm, T)
	MD,CurrentFile=initialize_calculation(Norm, T, dimension, numberFlag, CurrentFile);
	# MD = traditional_MTS(Norm, T, 0)
	# print "The mean of the MD values are", MD.mean()
	# print ("The mean of the distance values are", MD.mean())
	if par==1:
		return CurrentFile # np.resize(np.array(CurrentFile),(len(CurrentFile),dimension));
	else:
		return MD
##############################################################
def fisher(DM, M_1, M_2, M_3, M_4):
    # First let us decide on the prior probabilities
	pi1=1/float(4);
	pi2=1/float(4);
	pi3=1/float(4);
	pi4=1/float(4);
	for i in range(0,200):
		Prob=[];
		Prob_label= [];
		i = 0
		for element in DM:
			P_1 = ((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
			P_2 = ((1+(1/float(M_2)))*pi2*exp(-0.5*element[1])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
			P_3 = ((1+(1/float(M_3)))*pi3*exp(-0.5*element[2])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
			P_4 = ((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1]))+((1+(1/float(M_3)))*pi3*exp(-0.5*element[2]))+((1+(1/float(M_4)))*pi4*exp(-0.5*element[3])));
			Prob. append(P_1);
			Prob. append(P_2);
			Prob. append(P_3);
			Prob. append(P_4);
			lab_max = max(Prob)
			index_max= np.where( Prob == lab_max)
			Prob_label.append(index_max);
			Prob=[]
		Prob_label=np.array(Prob_label)
		c = [0,0,0,0];
		c[0]=list(Prob_label).count(0)
		c[1]=list(Prob_label).count(1)
		c[2]=list(Prob_label).count(2)
		c[3]=list(Prob_label).count(3)
		pi1=((c[0])/float(c[0] + c[1] + c[2] +c[3]))
		pi2=((c[1])/float(c[0] + c[1] + c[2] +c[3]))
		pi3=((c[2])/float(c[0] + c[1] + c[2] +c[3]))
		pi4=((c[3])/float(c[0] + c[1] + c[2] +c[3]))

	return np.resize(np.array(Prob_label),(len(Prob_label),))


##########################################################################
def classification_comparison(T,labels_T):

	names = ["Nearest Neighbors", "RBF SVM", "Decision Tree",
	         "Random Forest", "AdaBoost", "Naive Bayes", "LDA"]

	classifiers = [
	    KNeighborsClassifier(3),
	    SVC(gamma=2, C=1),
	    DecisionTreeClassifier(max_depth=5),
	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    LinearDiscriminantAnalysis()]

	Linearlyseparable =(T,labels_T);
	datasets= [ Linearlyseparable ]
	i = 1

	# iterate over datasets
	S =[]
	S1=[]

	for ds in datasets:

	    # preprocess dataset, split into training and test part
	    X, y = ds
	    X = StandardScaler().fit_transform(X)
	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
	    i += 1

	    # iterate over classifiers
	    for name, clf in zip(names, classifiers):
			clf.fit(X_train, y_train)
			S1.append(clf.score(X_test, y_test)*100)

			labels = clf. predict(X_test)
			index_1 = [i for i,v in enumerate(y_test) if v == 0 ]
			index_2 = [i for i,v in enumerate(y_test) if v == 1 ]

			L_0 = labels[index_1]
			L_1 = labels[index_2]

			# p_value =list(L_0).count(1)/float(len(L_0));
			p_value =list(L_1).count(0)/float(len(L_1));
			S.append(p_value)

	return S,S1




def comparison_dimred_classification(IR, OR, Norm, NL, Norm_T):
## Lets do the dimension reduction and classification comparison now.
# First Lets generate some data for two class classification problem
	Ref    = Norm
	Data_1 = NL
	Data_2 = OR
	X = np.concatenate((Data_1, Data_2))
	Y = np.concatenate((np.zeros(Data_1.shape[0]), (np.zeros(Data_2.shape[0])+1)))
	start_time = time.time()
	nc = 3
	Time = []
	# 1.  PCA as methods for dimension reduction
	print "PCA"
	start_time = time.time()
	print "The shape of the ref data is", Ref.shape
	pca = PCA(n_components=nc)
	pca.fit(X)
	CF1 = pca.transform(X)
	print "The time spent  for dimension reduction is", (time.time()-start_time)
	Time.append(time.time()-start_time)

	# 2. GDM-HDR as a method for dimension reduction method
	print "GDM-HDR"
	rand= [0 for i in xrange(Ref.shape[1])]
	for i in range(0,Ref.shape[1]):
			rand[i]= random.randint(0,(Ref.shape[1]-1))
	# Ref1 = Ref[:,rand]
	X1    = X[:,rand]
	Ref1   = Ref[:,rand]
	start_time = time.time()
	CF2 = Classification_Distances(Ref1, X1, 1, 111, nc)
	print "The time spent for dimension reduction is", (time.time()-start_time)
	Time.append(time.time()-start_time)


	# #  3. Random Projections
	# print "Random Projection"
	# rp  = SparseRandomProjection(n_components = nc)
	# start_time = time.time()
	# CF3  = rp.fit_transform(X)
	# print "The time spent for dimension reduction is", (time.time()-start_time)
	# Time.append(time.time()-start_time)


	# 4. ISOMAP
	print "ISOMAP"
	start_time = time.time()
	CF4 = manifold.Isomap(n_neighbors= 50, n_components=nc).fit_transform(X[0:1000,:])
	print "The time spent for dimension reduction is", (time.time()-start_time)
	Time.append(time.time()-start_time)

	# 5. Locally Linear Embedding
	print "LLE"
	start_time = time.time()
	CF5 = manifold.LocallyLinearEmbedding(n_neighbors=50, n_components=nc,
									eigen_solver='auto',
							method='standard').fit_transform(X)
	print "The time spent for dimension reduction is", (time.time()-start_time)
	Time.append(time.time()-start_time)

	# 6. Independent Component Analysis
	print "ICA"
	from sklearn.decomposition import FastICA
	start_time = time.time()
	est = FastICA(n_components= nc, whiten=True).fit(Ref)
	CF6  = est.transform(X)
	print "The time spent for dimension reduction is", (time.time()-start_time)
	Time.append(time.time()-start_time)

	dataset = [(CF1,Y), (CF2,Y), (CF4,Y[0:1000]), (CF5,Y), (CF6,Y)]

	# print "The shapes are", N.shape, N1.shape, CF.shape, Y.shape
	# classification_dimred(N,N1,CF,Y)
	i = 0
	for ds in dataset:
		CF,Y = ds
		Score, TypeOneError= classification_comparison(CF,Y)

		print "i == ", i
		i=i+1

		print "Score", format(float(Score[0]),'.2f'), '&', format(float(Score[1]),'.2f'), '&', format(float(Score[2]),'.2f'), '&', format(float(Score[3]),'.2f'), '&', format(float(Score[4]),'.2f'), '&', format(float(Score[5]),'.2f'), '&', format(float(Score[6]),'.2f')


		print "Score", format(float(TypeOneError[0]),'.2f'), '&', format(float(TypeOneError[1]),'.2f'), '&', format(float(TypeOneError[2]),'.2f'), '&', format(float(TypeOneError[3]),'.2f'), '&', format(float(TypeOneError[4]),'.2f'), '&', format(float(TypeOneError[5]),'.2f'), '&', format(float(TypeOneError[6]),'.2f')

		print "The averages"
		print sum(Score)/float(len(Score))
		print sum(TypeOneError)/float(len(TypeOneError))

	print "Time", format(float(Time[0]),'.4f'), '&', format(float(Time[1]),'.4f'), '&', format(float(Time[2]),'.4f'), '&', format(float(Time[3]),'.4f'), '&', format(float(Time[4]),'.4f');
		# print "The average efficiency is", sum(Score)/float(len(Score))
	print "The time spent is", (time.time()-start_time)
	return Time

##########################################################################
# Call all the Functions for analysis.
# First start with collecting samples
def Start_Analysis_Bearing():
	NL, IR, OR, N         = Collect_samples_Bearing(path, 1500)
	print(IR.shape,  OR.shape,  NL.shape,  N.shape)
	N = np.loadtxt('Norm.csv', delimiter = ',' )
	for i in range(0,100):
		print ('\n i == ', i)
		NL1= NL.copy()
		IR1= IR.copy()
		OR1= OR.copy()

		P =[]
		S =[]
		MD=[]

		### Lets go for IR
		P      =  IR1.copy()
		MD     =  Classification_Distances(IR,  np.array(P),0,111)
		p      =  10
		CF 	   =  []
		indexes=  list([ i for i,v in enumerate(MD) if v < p ]);
		CF.append([ P[i,:] for i in indexes ]);
		IR     =  np.array(CF[0]).copy()


		P =[]
		S =[]
		MD=[]

		### Lets go for OR
		P      =  OR1.copy()
		MD     =  Classification_Distances(OR,  np.array(P),0,111)
		p      =  10
		CF 	   =  []
		indexes=  list([ i for i,v in enumerate(MD) if v< p ]);
		CF.append([ P[j,:] for j in indexes ]);
		OR     =  np.array(CF[0]).copy()


		P =[]
		S =[]
		MD=[]

		### Lets go for NL
		P      =  NL1.copy()
		MD     =  Classification_Distances(NL,  np.array(P),0,111)
		p      =  10
		CF 	   =  []
		indexes=  list([ i for i,v in enumerate(MD) if v< p ]);
		CF.append([ P[j,:] for j in indexes ]);
		NL     =  np.array(CF[0]).copy()


		P =[]
		S =[]
		MD=[]



	np.savetxt(     'IR_sample.csv', IR, delimiter=',')
	np.savetxt(     'OR_sample.csv', OR, delimiter=',')
	np.savetxt(     'NL_sample.csv', NL, delimiter=',')
	IR = np.loadtxt('IR_sample.csv',     delimiter=',')
	OR = np.loadtxt('OR_sample.csv',     delimiter=',')
	NL = np.loadtxt('NL_sample.csv',     delimiter=',')

	print( IR.shape,  OR.shape,  NL.shape)

	return


def Collectsamples_test():
	IR = np.loadtxt('IR_sample.csv', delimiter=',')
	OR = np.loadtxt('OR_sample.csv', delimiter=',')
	NL = np.loadtxt('NL_sample.csv', delimiter=',')
	N  = np.loadtxt('N_sample.csv', delimiter=',')

	Test_NL, Test_IR, Test_OR, Test_N = Collect_samples_Bearing(path,1000)


	# NL
	print "NL"
	T = Test_NL.copy()
	N = NL.copy()
	MD = Classification_Distances(np.array(N),  np.array(T),0,111)
	p = 1
	CF = []
	indexes=list([i for i,v in enumerate(MD) if v  < p]);
	CF.append( [ T[i,:] for i in indexes ]);
	S= np.array(CF[0]).copy();
	print S.shape
	np.savetxt('NL_test.csv', S, delimiter=',')

	T=[]
	N=[]
	S=[]
	print "IR"
	T = Test_IR.copy()
	N = IR.copy()
	MD = Classification_Distances(np.array(N),  np.array(T),0,111)
	p = 1
	CF = []
	indexes=list([i for i,v in enumerate(MD) if v  < p]);
	CF.append( [ T[i,:] for i in indexes ]);
	S= np.array(CF[0]).copy()
	print S.shape
	np.savetxt('IR_test.csv', S, delimiter=',')



	# OR
	print "OR"
	T = Test_OR.copy()
	N = OR.copy()
	MD = Classification_Distances(np.array(N),  np.array(T),0,111)
	p = 1
	CF = []
	indexes=list([i for i,v in enumerate(MD) if v  < p]);
	CF.append( [ T[i,:] for i in indexes ]);
	S= np.array(CF[0])
	print S.shape
	np.savetxt('OR_test.csv', S, delimiter=',')



	# N
	print "N"
	T = Test_N.copy()
	N = N.copy()
	MD = Classification_Distances(np.array(N),  np.array(T),0,111)
	p = 1
	CF = []
	indexes=list([i for i,v in enumerate(MD) if v  < p]);
	CF.append( [ T[i,:] for i in indexes ]);
	S= np.array(CF[0])
	print S.shape
	np.savetxt('N_test.csv', S, delimiter=',')


def Classification_Faults_Bearing(N, T):
	p_value_0=[]
	p_value_1=[]
	p_value_2=[]
	p_value_3=[]

	for total in range(0,1):
		i = 0 ;

		for element in T:
			Test = element.copy()
			D_M= np.zeros((Test.shape[0],4))
			D_M[:,0]=np.resize(np.array(Classification_Distances( N[0] , Test,0,111,3)), (Test.shape[0],));
			D_M[:,1]=np.resize(np.array(Classification_Distances( N[1] , Test,0,111,3)), (Test.shape[0],));
			D_M[:,2]=np.resize(np.array(Classification_Distances( N[2] , Test,0,111,3)), (Test.shape[0],));
			D_M[:,3]=np.resize(np.array(Classification_Distances( N[3] , Test,0,111,3)), (Test.shape[0],));
			# D_M[:,0]=np.resize(np.array(traditional_MTS(Test, N[0],0)), (Test.shape[0],));
			# D_M[:,1]=np.resize(np.array(traditional_MTS(Test, N[1],0)), (Test.shape[0],));
			# D_M[:,2]=np.resize(np.array(traditional_MTS(Test, N[2],0)), (Test.shape[0],));
			# D_M[:,3]=np.resize(np.array(traditional_MTS(Test, N[3],0)), (Test.shape[0],));
			Label = fisher(D_M, N[0].shape[0], N[1].shape[0], N[2].shape[0], N[3].shape[0])

			# print ("The i is", i)
			# print ("The total number is", len(Label))
			c = [0,0,0,0];
			c[0]=list(Label).count(0)
			c[1]=list(Label).count(1)
			c[2]=list(Label).count(2)
			c[3]=list(Label).count(3)
			# print ("The Counter of the labels are", c)
			if i ==0:
				p_value_0.append((c[1] + c[2] +c[3])/float(c[0] + c[1] + c[2] +c[3]))
			if i ==1:
				p_value_1.append((c[0] + c[2] +c[3])/float(c[0] + c[1] + c[2] +c[3]))
			if i ==2:
				p_value_2.append((c[1] + c[0] +c[3])/float(c[0] + c[1] + c[2] +c[3]))
			if i ==3:
				p_value_3.append((c[1] + c[2] +c[0])/float(c[0] + c[1] + c[2] +c[3]))
			i = i+1;
			# print ("The actual Labels: (0) IR (1) OR (2) NL (3) N \n \n");
	P=[]
	P.append(sum(p_value_0)/float(len(p_value_0)))
	P.append(sum(p_value_1)/float(len(p_value_1)))
	P.append(sum(p_value_2)/float(len(p_value_2)))
	P.append(sum(p_value_3)/float(len(p_value_3)))
	return P

def Get_Num(Ref, Data):


	# detect = traditional_MTS(Ref, np.array(Data), 0)
	detect = Classification_Distances(Ref,  np.array(Data), 0, 111, 2)

	print "The mean of the MD values are", detect.mean()
	p =4.615
	index1=list([i for i,v in enumerate(detect) if v  > p]);
	p =6.42
	index2=list([i for i,v in enumerate(detect) if v  > p]);
	p =9.26
	index3=list([i for i,v in enumerate(detect) if v  > p]);
	print "p = 4.615; 0.1",  len(index1), (len(Data)-len(index1))
	print "p = 6.028; 0.05", len(index2), (len(Data)-len(index2))
	print "p =  9.26; 0.01",  len(index3), (len(Data)-len(index3))
	print len(Data)





def Classification(IR, OR, NL, N, Temp_IR, Temp_OR,Temp_NL, Temp_N):
	P_value =[]
	sample_size=500;
	rand = rand= [0 for i in xrange(sample_size)]
	for p in range(0,10):
		for i in range(0,sample_size):
			rand[i]= random.randint(0,(Temp_N.shape[0]-1))
		Test_N = Temp_N[rand,:];
		for i in range(0,sample_size):
				rand[i]= random.randint(0,(Temp_IR.shape[0]-1))
		Test_IR = Temp_IR[rand,:];
		for i in range(0,sample_size):
				rand[i]= random.randint(0,(Temp_OR.shape[0]-1))
		Test_OR = Temp_OR[rand,:];
		for i in range(0,sample_size):
			rand[i]= random.randint(0,(Temp_NL.shape[0]-1))
		Test_NL = Temp_NL[rand,:];
		T=[]
		T.append(Test_IR)
		T.append(Test_OR)
		T.append(Test_NL)
		T.append(Test_N)
		np.array(T)
		# np.resize(np.array(T), (4,len(T),11));
		N= []
		N.append(IR)
		N.append(OR)
		N.append(NL)
		N.append(Norm)
		np.array(N)
		# np.resize(np.array(N), (4,len(N),11));
		print "The top iterations", p
		P_value.append(Classification_Faults_Bearing(N,T))
		P = np.array(P_value)
		print "The p value for the IR is", (1-sum(P[:,0])/len(P[:,0]))
		print "The p value for the OR is", (1-sum(P[:,1])/len(P[:,1]))
		print "The p value for the NL is", (1-sum(P[:,2])/len(P[:,2]))
		print "The p value for the N is" , (1-sum(P[:,3])/len(P[:,3]))
# np.savetxt( 'N_test.csv', Data, delimiter=',')


def classification_dimred(N, N1, Test, Y):

	D_M= np.zeros ((Test.shape[0],2))
	D_M[:,0]=np.resize(np.array(traditional_MTS(Test,  N)), (Test.shape[0],));
	D_M[:,1]=np.resize(np.array(traditional_MTS(Test, N1)), (Test.shape[0],));
	M_1=   N.shape [0]
	M_2 =  N1.shape[0]
	print N.shape
	print N1.shape
	print Test.shape
	# First let us decide on the prior probabilities
	pi1=1/float(2);
	pi2=1/float(2);
	for i in range(0,1):
		Prob_label= [];
		i = 0
		for element in D_M:
			P_1 = ((1+(1/float(M_1)))*pi1*exp(-0.5*element[0])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0]))   	+((1+(1/float(M_2)))*pi2*exp(-0.5*element[1])));
			P_2 = ((1+(1/float(M_2)))*pi2*exp(-0.5*element[1])) / (((1+(1/float(M_1)))*pi1*exp(-0.5*element[0]))    +((1+(1/float(M_2)))*pi2*exp(-0.5*element[1])));

			lab_max = max(P_1,P_2)
			if lab_max==P_1:
				Prob_label.append(0);
			else:
				Prob_label.append(1);

		P=np.resize(np.array(Prob_label),(len(Prob_label),))
		#print Prob_label
		c= Counter(Prob_label)
		pi1=((c[0])/float( c[0] + c[1] ))
		pi2=((c[1])/float( c[0] + c[1] ))
	print ("The Counter of the labels are", Counter(P))
	print ("The Counter of the labels are", Counter(Y))
	c1= Counter(Y)
	Corr= P[0:c1[0]]
	TE  = Y[0:c1[0]]
	c= Counter(Corr)
	print (c[0]/float(c1[0]))*100



def Normal():
	from copulalib.copulalib import Copula
	deviation=5
	N_points=10000;
	N_Features=200;
	size_init_data=100;
	Data_array_Ref=np.zeros((N_points,N_Features));
	Data_array_Test=np.zeros((N_points,N_Features));
	# Let us create the numpy array
	for i in range(0,N_Features):
		# Generate random (normal distributed) numbers::
		x = np.random.normal(size=size_init_data);
		y = (x)+np.random.normal(size=size_init_data);

		# Make the instance of Copula class with x, y and clayton family::
		foo = Copula(x, y, family='clayton');
		X1, Y1 = foo.generate_xy(N_points);

		#print X1.shape
        #print Y1.shape
		print "The iteration going on is", i;
		X2=X1+deviation;
		Y2=Y1+deviation;

		# plot_copula(X1,Y1,X2,Y2,'Copula_parabola_T')
        Data_array_Ref[:,i]=X1.copy();
        Data_array_Ref[:,i]=Y1.copy();
        Data_array_Test[:,i]=X2.copy();
        Data_array_Test[:,i]=Y2.copy();
        i=i+1;
	np.savetxt("foo_Ref.csv", Data_array_Ref, delimiter=",");
	np.savetxt("foo_Test.csv", Data_array_Test, delimiter=",");


def Fault_trend(T):
	N    = np.loadtxt( 'N_sample.csv', delimiter=",")
	CF   = Classification_Distances(N, T,0,111)
	return CF


def Classify_Total(Test, IR, OR, Norm, NL):
	D_M= np.zeros((Test.shape[0],4))
	D_M[:,0]=np.resize(np.array(Classification_Distances( IR   , Test,0,111,3)),(Test.shape[0],));
	D_M[:,1]=np.resize(np.array(Classification_Distances( OR   , Test,0,111,3)), (Test.shape[0],));
	D_M[:,2]=np.resize(np.array(Classification_Distances( NL   , Test,0,111,3)), (Test.shape[0],));
	D_M[:,3]=np.resize(np.array(Classification_Distances( Norm , Test,0,111,3)), (Test.shape[0],));
	# D_M[:,0]=np.resize(np.array(traditional_MTS(Test, IR,0  )), (Test.shape[0],));
	# D_M[:,1]=np.resize(np.array(traditional_MTS(Test, OR,0  )), (Test.shape[0],));
	# D_M[:,2]=np.resize(np.array(traditional_MTS(Test, NL,0  )), (Test.shape[0],));
	# D_M[:,3]=np.resize(np.array(traditional_MTS(Test, Norm,0)), (Test.shape[0],));
	Label = fisher(D_M, IR.shape[0], OR.shape[0], NL.shape[0], Norm.shape[0])
	# print ("The i is", i)
	# print ("The total number is", len(Label))
	c = [0,0,0,0];
	c[0]=list(Label).count(0)
	c[1]=list(Label).count(1)
	c[2]=list(Label).count(2)
	c[3]=list(Label).count(3)
	print c
	return c, Label


def class_timeseries_start(Data, IR, OR, Norm, NL):

	c, label = Classify_Total(Data, IR, OR, Norm, NL)
	print (c[0]/float(Temp_IR.shape[0])), (c[1]/float(Temp_OR.shape[0])), (c[2]/float(Temp_NL.shape[0])), (c[3]/float(Norm.shape[0]))



	T1 = label[0: (Temp_IR.shape[0])]
	T2 = label[(Temp_IR.shape[0]):((Temp_IR.shape[0])+(Temp_OR.shape[0]))]
	T3 = label[((Temp_IR.shape[0])+(Temp_OR.shape[0])) : (  ((Temp_IR.shape[0])+(Temp_OR.shape[0]))   +(Temp_NL.shape[0]))]
	T4 = label[(  ((Temp_IR.shape[0])+(Temp_OR.shape[0]))   +(Temp_NL.shape[0])):len(label)]

	print ((list(T1).count(0))/float(len(T1)))
	print ((list(T2).count(1))/float(len(T2)))
	print ((list(T3).count(2))/float(len(T3)))
	print ((list(T4).count(3))/float(len(T4)))

def RollingDataImport():
	# Start_Analysis_Bearing()
	IR    	=  np.loadtxt('IR_sample.csv', delimiter=',')
	OR    	=  np.loadtxt('OR_sample.csv', delimiter=',')
	NL    	=  np.loadtxt('NL_sample.csv', delimiter=',')
	Norm  	=  np.loadtxt('Norm.csv'     , delimiter=',')

	sheet    = 'Test';
	f        = 'IR1.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'Test';
	f        = 'OR1.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'Test';
	f        = 'NL1.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));

	sheet    = 'normal';
	f        = 'Normal_1.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));


	return Temp_Norm, Temp_IR, Temp_OR, Temp_NL, IR, NL, OR, Norm


def generate_new_data(n_sam, n_fea, n_inf):
	X,y = make_classification(n_samples=n_sam, n_features=n_fea, n_informative=n_inf, n_redundant=(n_fea-n_inf), n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01,class_sep=2.0, hypercube=True, shift=10.0, scale=1.0, shuffle=True, random_state= 9000)


	index_1 = [i for i,v in enumerate(y) if v == 0 ]

	index_2 = [i for i,v in enumerate(y) if v == 1 ]

	Data_class_1 = X[index_1,:]
	L1 = y[index_1];
	L2 = y[index_2];
	Data_class_2 = X[index_2,:]
	X_train_1, X_test_1 = train_test_split(Data_class_1, test_size=.5)
	X_train_2, X_test_2 = train_test_split(Data_class_2, test_size=.5)

	np.savetxt('Train_1.csv', X_train_1, delimiter=',')
	np.savetxt('Train_2.csv', X_test_1, delimiter=',')
	np.savetxt('Test_1.csv', X_test_2, delimiter=',')
	np.savetxt('Total_Test.csv', X, delimiter=',')
	np.savetxt('Total_labels.csv', y, delimiter=',')

def test_data_samples(Ref, dat):
	t= time.time()
	CF = Classification_Distances(Ref, data, 0, 111)
	print "The time for GDM-HDR is", (time.time()-t)
	print "The length of the values are", len(CF)
	count = 0 ;
	for element in CF:
		if element < 10 :
			count = count+1
	print count;
	print "The mean of the distance values are", CF.mean()
	t= time.time()
	CF=[]
	CF = traditional_MTS(Ref, data, 0)
	print "The time fo MD is", (time.time()-t)
	print "The mean of the distance values are", CF.mean()
	count = 0 ;
	for element in CF:
		if element < 10 :
			count = count+1
	print count;

def comparison_dimred_Artificial_Dataset(Ref, X, Y, nc):
	## Lets do the dimension reduction and classification comparison now.


		# 1. Random Forest
		# CF= np.zeros((X.shape[0], 2))
		# from sklearn.ensemble import ExtraTreesClassifier
		# forest = ExtraTreesClassifier(n_estimators=250,
		#                               random_state=0)
		# forest.fit(X, Y)
		# importances = forest.feature_importances_
		# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
		#              axis=0)
		# indices = np.argsort(importances)[::-1]
		# CF  [:,0]= X[:,indices[0]]
		# CF  [:,1]= X[:,indices[1]]
		# print "The time spent is", (time.time()-start_time)
		#
		# indices = np.argsort(importances)[::-1]

		# # Print the feature ranking
		# print("Feature ranking:")
		#
		# for f in range(X.shape[1]):
		# 	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
		# 	# Plot the feature importances of the forest
		# 	plt.figure()
		# 	plt.title("Feature importances")
		# 	plt.bar(range(X.shape[1]), importances[indices],
	    #    	color="r", yerr=std[indices], align="center")
		# 	plt.xticks(range(X.shape[1]), indices)
		# 	plt.xlim([-1, X.shape[1]])
		# 	plt.show()
		# return

		Time = []

		# 1.  PCA as methods for dimension reduction
		print "PCA"
		start_time = time.time()
		print "The shape of the ref data is", Ref.shape
		pca = PCA(n_components=nc)
		pca.fit(Ref)
		CF1 = pca.transform(X)
		print "The time spent  for dimension reduction is", (time.time()-start_time)
		Time.append(time.time()-start_time)

		# 2. GDM-HDR as a method for dimension reduction method
		print "GDM-HDR"
		rand= [0 for i in xrange(Ref.shape[1])]
		for i in range(0,Ref.shape[1]):
				rand[i]= random.randint(0,(Ref.shape[1]-1))
		# Ref1 = Ref[:,rand]
		X1    = X[:,rand]
		Ref1   = Ref[:,rand]
		start_time = time.time()
		CF2 = Classification_Distances(Ref1, X1, 1, 111, nc)
		print "The time spent for dimension reduction is", (time.time()-start_time)
		Time.append(time.time()-start_time)

		# 4. ISOMAP
		print "ISOMAP"
		start_time = time.time()
		CF4 = manifold.Isomap(n_neighbors= 50, n_components=nc).fit_transform(X)
		print "The time spent for dimension reduction is", (time.time()-start_time)
		Time.append(time.time()-start_time)

		# 5. Locally Linear Embedding
		print "LLE"
		start_time = time.time()
		CF5 = manifold.LocallyLinearEmbedding(n_neighbors=50, n_components=nc,
                                        eigen_solver='auto',
                                method='standard').fit_transform(X)
		print "The time spent for dimension reduction is", (time.time()-start_time)
		Time.append(time.time()-start_time)

		# 6. Independent Component Analysis
		print "ICA"
		from sklearn.decomposition import FastICA
		start_time = time.time()
		est = FastICA(n_components= nc, whiten=True).fit(X[0:600,:])
		CF6  = est.transform(X)
		print "The time spent for dimension reduction is", (time.time()-start_time)
		Time.append(time.time()-start_time)

		dataset = [(CF1,Y), (CF2,Y), (CF4,Y), (CF5,Y), (CF6,Y)]

		# print "The shapes are", N.shape, N1.shape, CF.shape, Y.shape
		# classification_dimred(N,N1,CF,Y)
		i = 0
		S = []
		T = []
		for ds in dataset:
			CF,Y = ds
			Score, TypeOneError= classification_comparison(CF,Y)

			print "i == ", i
			i=i+1

			print "Score", format(float(Score[0]),'.2f'), '&', format(float(Score[1]),'.2f'), '&', format(float(Score[2]),'.2f'), '&', format(float(Score[3]),'.2f'), '&', format(float(Score[4]),'.2f'), '&', format(float(Score[5]),'.2f'), '&', format(float(Score[6]),'.2f')


			print "Score", format(float(TypeOneError[0]),'.2f'), '&', format(float(TypeOneError[1]),'.2f'), '&', format(float(TypeOneError[2]),'.2f'), '&', format(float(TypeOneError[3]),'.2f'), '&', format(float(TypeOneError[4]),'.2f'), '&', format(float(TypeOneError[5]),'.2f'), '&', format(float(TypeOneError[6]),'.2f')

			print "The averages"
			S.append(sum(Score)/float(len(Score)))
			T. append(sum(TypeOneError)/float(len(TypeOneError)))
		print "Time", format(float(Time[0]),'.4f'), '&', format(float(Time[1]),'.4f'), '&', format(float(Time[2]),'.4f'), '&', format(float(Time[3]),'.4f'), '&', format(float(Time[4]),'.4f');
			# print "The average efficiency is", sum(Score)/float(len(Score))
		print "The time spent is", (time.time()-start_time)
		return S,T

def Test_Computation():
	rand= [0 for i in xrange(200)]
	for i in range(0,200):
			rand[i]= random.randint(11,2000)
	P = sorted(rand, key=int)
	print P
	time_total = []
	# Let us work with another data-set now
	for n_fea in P:

		n_sam = 1000
		n_inf = 10
		n_ret = 5

		generate_new_data(n_sam, n_fea, n_inf)
		Ref          = np.loadtxt('Train_1.csv',      delimiter = ',')
		Total_Test   = np.loadtxt('Total_Test.csv',   delimiter = ',')
		Labels 		 = np.loadtxt('Total_labels.csv', delimiter = ',')

		time_total.append(comparison_dimred_Artificial_Dataset(Ref, Total_Test, Labels, n_ret))
	np.savetxt('Time.csv', time_total ,delimiter=',')
	np.savetxt('Index.csv', P ,delimiter=',')
