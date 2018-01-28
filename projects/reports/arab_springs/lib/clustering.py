import pickle
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, sparse
import scipy.sparse.linalg
import scipy
from sklearn.cluster import KMeans
from pygsp import graphs, filters, plotting
import operator
import io
from lib import models, graph, coarsening, utils
get_ipython().magic('matplotlib inline')

def take_eigenvectors(laplacian, K=5):
	eigenvalues, eigenvectors =  sparse.linalg.eigsh(laplacian, k=K, which = 'SA')
	return eigenvalues, eigenvectors

def do_kmeans(eigenvectors, K=5):
	#kmeans to find clusters
	kmeans = KMeans(n_clusters=K, random_state=0).fit(eigenvectors)
	return kmeans.labels_

def label_data(df, kmeans_labels, K=5, NUMBER = 40):
	counts = [dict() for x in range(K)]
	for i, label in enumerate(kmeans_labels):
		words = df.loc[i].Tokens
		for w in words:
			try: 
				counts[label][w]+=1
			except: counts[label][w]=1 
	total = {}
	for k in range(K):
		sorted_words = sorted(counts[k], key=operator.itemgetter(1), reverse=True)[:NUMBER]
		for w in sorted_words:
			try:
				total[w]+=1
			except: total[w]=1
	labels = [[] for i in range(K)]
	for k in range(K):
		sorted_words = sorted(counts[k], key=operator.itemgetter(1), reverse=True)[:NUMBER]
		for w in sorted_words: 
			if total[w]==1: labels[k].append(w)
	return labels