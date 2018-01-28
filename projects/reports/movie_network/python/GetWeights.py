import sys, os, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
def GetWeights(distances,NEIGHBORS):
    
    kernel_width = distances.mean()
    print(kernel_width)
    #getting the weights using given kernel
    weights = np.zeros((len(distances),len(distances)))
    for i in range(0, len(distances)):
        for j in range(0,len(distances)):
            weights[i][j]=  np.exp((-distances[i][j]*distances[i][j])/(kernel_width*kernel_width))
            
    #setting diagonal terms to zeros
    np.fill_diagonal(weights, 0)
    fix, axes = plt.subplots(2, 2, figsize=(17, 8))
    def plot(weights, axes):
        axes[0].spy(weights)
        axes[1].hist(weights[weights > 0].reshape(-1), bins=50);
    plot(weights, axes[:, 0])
    newweights = np.zeros((len(distances),len(distances)))
    #dropping edges while keeping only 100 strongest
    #dropping per edge side to avoid disconected nodes
    counter=0
    for line in weights:
        ordered_indices = np.argsort(line)[len(line)-NEIGHBORS:]#[:len(line)-NEIGHBORS]
        for i in ordered_indices:
            newweights[counter][i] = line[i]
            newweights[i][counter] = line[i]
            line[i]=0
        counter+=1    

    weights = newweights
    #symetrizing weights matrix, needed due to case when 
    #edge between nodes A and B is in 100 strongest edges of A but not in 100 stronges edges of B
    bigger = weights.transpose() > weights
    weights = weights - weights*bigger + weights.transpose()*bigger
    plot(weights, axes[:, 1])
    return weights