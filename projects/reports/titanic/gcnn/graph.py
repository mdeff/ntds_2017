import numpy as np
import scipy as sp
import scipy.sparse, scipy.linalg
import networkx as nx


def grid_coordinates(n: int):
    """
    Coordinates (x, y) list for a grid of size n x n.
    """
    idx = np.arange(n)
    return np.reshape(np.meshgrid(idx, idx), (2, -1)).T


def knn(z, k: int = 4, metric: str = 'euclidean'):
    """
    K-NN graph from list of features. Might return more than k neighbors in case of distance equality.
    """
    dists = sp.spatial.distance.pdist(z, metric)
    dists = sp.spatial.distance.squareform(dists)
    
    weights = np.exp(- dists ** 2 / 2)
    mask = weights < np.sort(weights, axis=1)[:, -k]
    weights[mask & mask.T] = 0
    
    assert np.all(weights == weights.T)
    return nx.from_numpy_array(weights > 0)


def knn3d(z, k: int = 4, metric: str = 'euclidean', d: int = 2):
    """
    K-NN hraph from list of features. Might return more than k neighbors in case of distance equality.
    """
    levels = [np.c_[z, np.ones_like(z) * d] for d in range(d)]
    return knn(np.concatenate(levels), k=k, metric=metric)


def kwraps(n: int, kd: int = 1):
    """
    Graph from a wrapped grid (border touch other borders) within kd elements. Not optimized.
    """
    
    g = nx.empty_graph()

    def add_edge(x1, y1, x2, y2, v=1):
        g.add_edge((x1, y1), (x2, y2))   

    for x in range(n):
        for y in range(n):
            for dx in range(-kd, kd + 1):
                for dy in range(-kd, kd + 1):
                    if dx != 0 or dy != 0:
                        add_edge(x, y, (x + dx) % n, (y + dy) % n)  
    
    return g


def kwraps3d(n: int, kd: int = 1, d: int = 2):
    """
    Graph from a wrapped 3d grid (border touch other borders) within kd elements. Not optimized.
    """
    
    g = nx.empty_graph()

    def add_edge(x1, y1, z1, x2, y2, z2, v=1):
        g.add_edge((z1, x1, y1), (z2, x2, y2))   
        
    for x in range(n):
        for y in range(n):
            for z in range(d):
                for dx in range(-kd, kd + 1):
                    for dy in range(-kd, kd + 1):
                        for dz in range(-kd, kd + 1):
                            if dx != 0 or dy != 0 or dz != 0:
                                add_edge(x, y, z, (x + dx) % n, (y + dy) % n, (z + dz) % d)  
    
    return g


def remove_random_edges(graph, size):
    """
    Remove some edges from graph.
    """
    edges = list(graph.edges)
    
    for edge in np.random.choice(range(len(edges)), size=size, replace=False):
        edge = edges[edge]
        graph.remove_edge(edge[0], edge[1])
    
    return graph


def fourier(laplacian):
    """
    Graph fourier basis for a laplacian using SVD.
    """
    return sp.linalg.svd(laplacian)[0]

