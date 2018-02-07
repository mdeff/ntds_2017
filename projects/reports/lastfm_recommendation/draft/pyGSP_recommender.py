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
import pygsp as gsp
from Dataset import Dataset
import pickle

plt.rcParams['figure.figsize'] = (10, 5)
gsp.plotting.BACKEND = 'matplotlib'


#%% IMPORT DATA
data = Dataset()
data.prune_ratings()
data.prune_friends()
data.split(test_ratio=0.17, seed=2)
data.normalize_weights()
friend_friend = data.build_friend_friend()
user_art = data.build_art_user().T

#%% ADD HIGH FREQUENCY CONTENT
np.random.seed(2)
rand = np.random.uniform(1,5, size=(user_art.size - np.count_nonzero(user_art)))
user_art[user_art==0] = rand

#%% ALGO
G = gsp.graphs.Graph(friend_friend)
G.estimate_lmax()
G.set_coordinates()
f = gsp.filters.expwin.Expwin(G)

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

pred_user_art = f.filter(user_art)
evaluate(pred_user_art, data)
