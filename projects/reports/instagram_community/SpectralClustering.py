
# coding: utf-8

# In[1]:

import numpy as np
import scipy as sc
from sklearn.cluster import KMeans


# In[2]:

def SpectralClustering (W, Nc, lap_type='unnormalized', Nc_max=0):
    r"""Estimate labels of the dataset whose weight matrix is given
        Parameters
        ----------
        W : ndarray
            The weight matrix which encodes the graph.
        lap_type : 'unnormalized', 'normalized'
            The type of Laplacian to be computed (default is 'unnormalized').
        Nc : number of clusters
        Nc_max : if Nc_max is given, decide Nc by searching for the eigengap on lowest Nc number of eigenvalue
        normalize_rows : bool
            After getting eigenvector marix, normalize each row if True
        Returns
        -------
        L : Laplacian matrix
        labels : label of each data point in terms of their cluster numbers
    """
    
    deg = W.sum(0)
    Lcomb = sc.sparse.csc.csc_matrix(np.diag(deg) - W) #combinatorial laplacian
    if lap_type == 'unnormalized':
        L = Lcomb
    else:
        D_2 = np.diag(1/deg**(1/2))
        L = sc.sparse.csc.csc_matrix(D_2 @ Lcomb @ D_2)
        
    # Eigendecomposition
    if Nc_max != 0:
        evals, evecs = sc.sparse.linalg.eigsh(L, Nc_max, which= 'SA')
        Nc = np.argmax(np.diff(evals)) +1 #eigengap proposition
        print('Number of clusters is determined as',Nc,'by the eigengap heuristic.')
        evecs = evecs[:,:Nc]
    else:
        evals, evecs = sc.sparse.linalg.eigsh(L, Nc, which= 'SA')
        
    if lap_type == 'unnormalized':
        embedding = evecs[:,1:] #discard constant eigenvector
    else:
        embedding = evecs
    kmeans = KMeans(n_clusters=Nc, random_state=0).fit(embedding)
    labels = kmeans.labels_
    return L, embedding, labels

