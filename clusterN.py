from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist, mahalanobis  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

# Evaluate the distance matrix,
def cdist_sparse( X, Y, clusters, metric, p):
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, metric, p)
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, metric, p ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), metric, p) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), metric, p ) [0]
    return d
# Random sample
def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]
# Create initial clusters
def initial_samples(X, N):
    C = {}
    for i in xrange(N):
        C[i] = randomsample(X,100)
    return C

#...............................................................................
def kmeans( X, centres, clusters = None, delta = 0.001, maxiter=100, metric="euclidean", p=2, verbose=1 ):

    if not issparse(X):
        X = np.asanyarray(X)
    centres = centres.todense() if issparse(centres) \
        else centres.copy()

    N, dim = X.shape
    k, cdim = centres.shape

    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))

    if verbose:
        print "kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric)

    allx = np.arange(N)
    prevdist = 0

    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centres, clusters, metric=metric, p=p )  # |X| x |centres|

        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50.astype(int)
        print "kmeans: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances




def nearestcentres( X, centres, metric="euclidean", p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        self.centres, self.Xtocentre, self.distances = kmeans(X, centres, **kwargs )
    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................
def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    Y = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        y = (np.zeros(int(n))+i)
        X.extend(x)
        Y.extend(y)
    X = np.array(X)[:N]
    Y = np.array(Y)[:]
    return X, Y

if __name__ == "__main__":
    # INITIAL DEFINITIONS
    n_sam = 10000
    ncluster = 10
    n_fea = (10+(2*ncluster))
    n_inf =  2*ncluster
    kmdelta = .001
    kmiter = 10
    from sklearn.datasets import  make_classification
    for m in ['cityblock', 'euclidean', 'mahalanobis' ]:
        # GENERATE DATA
        X,y = make_classification(n_samples = n_sam, n_features = n_fea, n_informative = n_inf, n_redundant = (n_fea-n_inf), n_classes = ncluster, n_clusters_per_class = 1, class_sep = 10, hypercube = True, shuffle = True, random_state = 9000)
        # INITIALIZE CLUSTERS
        clusters= initial_samples(X, ncluster)
        randomcentres = [ clusters[i].mean(axis=0) for i in xrange(len(clusters))]
        # We got the centers and the clustersm perform the K-means now.
        centres, label, dist = kmeans( X, np.array(randomcentres), clusters, \
        maxiter=kmiter, metric=m, verbose=2 )
        from sklearn import metrics
        print('%.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
                  %  (metrics.homogeneity_score(y, label),
                     metrics.completeness_score(y, label),
                     metrics.v_measure_score(y, label),
                     metrics.adjusted_rand_score(y, label),
                     metrics.adjusted_mutual_info_score(y, label),
                     metrics.silhouette_score(X, label,
                                              metric='euclidean',
                                              sample_size=X.shape[0])))
