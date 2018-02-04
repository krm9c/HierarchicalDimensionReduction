# First of all the imports 
# lets start with some imports
import os, sys
import numpy as np
from tqdm import tqdm

from Helper_func import classification_comparison, dim_reduction_comparison
# Classification Test
def classification_test():
    dataset = ['notmnist']
    param = [ [2, 168,'corr']]
    # dataset = ['arcene']
    # param =[[20, 25,'corr']]
    for i, element in enumerate(dataset):
        print("element", element, param[i])
        classification_comparison(element, param[i], 2)

def dim_reductionTest():
    # Dimension Reduction methods Test
    dataset = ['sensorless']
    param = [20]
    from Helper_func import dim_reduction_comparison
    for i, element in enumerate(dataset):
        print("element", element)
        dim_reduction_comparison( element, n_comp = param[i]) 
dim_reductionTest()