# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:26:10 2018

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
from sklearn.neighbors import NearestNeighbors
import pygsp as gsp
from Dataset import Dataset
import pickle

plt.rcParams['figure.figsize'] = (10, 5)
gsp.plotting.BACKEND = 'matplotlib'

#%% IMPORT DATA
data = Dataset()
data.prune_ratings()
data.prune_friends()
data.normalize_weights()

#%% CONSTANTS
TRESHOLD = 5    # Treshold of user_user connection strength to keep
K = 20

#%% BUILD ARTIST GRAPH EXAMPLE
# Build example graph
"""
G = gsp.graphs.Community(N=data.nart, Nc=K, seed=42)
art_tags = G.W.todense()

# Use the cosine distance
distances = sci.spatial.distance.pdist(art_tags, metric='cosine')
distances = sci.spatial.distance.squareform(distances)

# Add simplest kernel to go from distance to similarity measure
art_art = 1 - distances

"""
with open('art_art.pickle', 'rb') as fp:
    art_art = pickle.load(fp)
# Remove self-loops
np.fill_diagonal(art_art, 0)

#%% SPECTRAL CLUSTERING
algo = SpectralClustering(n_clusters=K, affinity='precomputed')
genres = algo.fit_predict(art_art)

#%% WEIGHTS EMBEDDING
user_genre = np.zeros((data.nuser, K))
for index, row in data.ratings.iterrows():
    upos = data.get_userPOS(row.userID)
    # Find the genre label associated to the artists in the selected row
    gpos = genres[ data.get_artistPOS(row.artistID) ]
    user_genre[upos,gpos] += row.weight
    
#%% FIND USERS CONNECTIONS
algo = NearestNeighbors(n_neighbors=5, metric='cosine')
algo.fit(user_genre)
user_user = algo.kneighbors_graph(user_genre).todense().A