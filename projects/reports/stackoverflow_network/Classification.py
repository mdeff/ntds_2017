import networkx as nx
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def classify_users(directed_network):
    """ Build unsupervised classifier with:
        In degree
        Out degree
        Avg Question votes
        Avg Answer votes
        Number accepted
    """

    features = create_features(directed_network)
    
    X = feature_engineering(np.array(features.as_matrix()))
    
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    print ("Label statistics:")
    print (np.unique(labels, return_counts=True))
      
    # Print classification in 2d area
    plt.scatter(X[:,1], X[:,2], c=labels, cmap='Set1')
    plt.xlabel('In degree')
    plt.ylabel('Out degree')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.show()
    
    plt.scatter(X[:,0], X[:,3], c=labels, cmap='Set1')
    plt.xlabel('Avg question votes')
    plt.ylabel('Avg answer votes')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.show()
    
    plt.scatter(X[:,1], X[:,3], c=labels, cmap='Set1')
    plt.ylabel('In degree')
    plt.xlabel('Avg answer votes')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.show()
    
    plt.scatter(X[:,0], X[:,2], c=labels, cmap='Set1')
    plt.ylabel('Out degree')
    plt.xlabel('Avg question votes')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.show()
    
    print_class_averages(features ,labels)
    
def print_class_averages(X,Y):
    
    X["label"] = Y
    print(X.groupby('label').mean())

def feature_engineering(X):
    
    X_new = X.copy()
    
    for i in range(X.shape[1]):
        # gained insight: features are distributed by power law
        #plt.hist(X[:,i])
        #plt.show()
        
        # log scale not possible due to zeros
        # X_new = np.log(X_new)
        
        # extended feature vectors
        poly = PolynomialFeatures(degree = 2, include_bias=False)
        # Took too long
        #X_new = poly.fit_transform(X_new)

        # normalize
        X_new = preprocessing.scale(X_new)
        
    return X_new

def create_features(network):
    """ Extract features from graph """
    
    nodes = network.nodes()    
    
    features = []
    for node in nodes:
        features_node =  {}
        
        in_edges = network.in_edges([node], data=True)
        out_edges = network.out_edges([node], data=True)
        
        features_node["degree_in"] =  len(in_edges)
        features_node["degree_out"] = len(out_edges)
        
        features_node["q_votes"] = 0
        for out_edge in out_edges:
            features_node["q_votes"] += out_edge[2]["votes_q"]
        if len(out_edges) != 0:
            features_node["q_votes"] /= len(out_edges)
        
        features_node["a_votes"] = 0
        for in_edge in in_edges:
            features_node["a_votes"] += in_edge[2]["votes_a"]
        if len(in_edges) != 0:
            features_node["a_votes"] /= len(in_edges)
            
        features.append(features_node)
        
    return pd.DataFrame.from_dict(features)