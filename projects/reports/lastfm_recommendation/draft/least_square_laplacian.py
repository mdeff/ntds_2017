# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:39:55 2018

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import pandas as pd
import os.path
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import surprise
import pygsp as gsp
from Dataset import Dataset
import pickle
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (10, 5)
gsp.plotting.BACKEND = 'matplotlib'


#%% IMPORT DATA
data = Dataset()
data.prune_ratings()
data.prune_friends()
data.split(test_ratio=0.17, seed=2)
data.normalize_weights()
friend_friend = data.build_friend_friend()

#%% CONSTRUCT SIGNAL
user_art = data.build_art_user().T

#%% ALGO

def lapLL(L, user_art, alpha=0.01): #rmse=2.18
    """ Laplacian Least Square: find prediction using a Least Square
        regularized over the graph """
    pred_user_art = np.zeros(user_art.shape)
    for art, y in enumerate(tqdm(user_art.T)):
        mask = y!=0
        M = np.diag(mask.astype('int'))
        pred_user_art[:,art] = np.linalg.solve(M + alpha*L, M@y)
    
    return pred_user_art

def evaluate(pred, data):
    """ Evaluate predictions with rmse score """
    loss = 0
    for u,a,r in data.test.itertuples(index=False):
        loss += (pred[data.get_userPOS(u), data.get_artistPOS(a)] - r)**2
        
    # Return a rmse of the predictions considering ratings 0-4
    # The multiplication factor of 4 is to be aligned with the rmse scores
    # of the netflix prixe which became popular over the years
    rmse = np.sqrt(loss / len(data.test))
    print('RMSE: ',rmse)
    return rmse

G = gsp.graphs.Graph(friend_friend)
G.compute_laplacian(lap_type='normalized')
pred_user_art = lapLL(G.L, user_art, alpha=0.01)
rmse = evaluate(pred_user_art, data)
