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

# Evaluate the density
def density(sample, ele, pdf):
    # moment_3 = moments(sample,3)
    # moment_4 = moments(sample,4)
    # moment_1 = moments(sample,1)
    # moment_2 = moments(sample,2)
    # moment_5 = moments(sample,5)
    # h1 = np.sum(moment_1)
    # h2 = np.sum(moment_2-1)
    # h3 = np.sum(moment_3-3*moment_1)
    # h5 = np.sum(moment_5-10*moment_3+15*moment_1)
    # B = sample.shape[0]
    # Check this out through math, edgeworth expansion for multivariate densities
    # factor = np.sum( 1+((moment_3*h2)/float(6*sqrt(B)))+\
    #     (((moment_4-3)*h3)/float(24*B))  +\
    #     ((np.power(moment_3,2)*h2)/float(72*B*B)))
    return np.exp(pdf.score_samples(ele))

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

## Lets do this for all data-sets and how this works
data_name = ['rolling', 'sensorless']

for data in data_name:
    from Helper_func import import_pickled_data
    XTr,yTr,TTe,yTe = import_pickled_data(data)
    
    from sklearn.model_selection import train_test_split
    XTr, XTe, yTr, yTe  = train_test_split( TTe, yTe,\
    test_size=0.30, random_state=42)
    

    ## Lets reduce the data first 


    print("Shapes", XTr.shape, XTe.shape, yTr.shape, yTe.shape)
    pdf = classify_samples(XTr, yTr)
    print(len(pdf))
    predict =  bayes_classifier(XTe, pdf)

    from sklearn.metrics import accuracy_score
    print accuracy_score(yTe, predict, normalize=True, sample_weight=None)

