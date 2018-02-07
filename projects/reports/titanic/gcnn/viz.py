from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import networkx as nx
import scipy as sp
import scipy.sparse

def plot_bands(e, size=75):
    """
    Plots the two bands side by side.
    """
    plt.title('band 1 | band_2')
    plt.imshow(np.c_[e.band_1.reshape(size, size), e.band_2.reshape(size, size)])

    
def plot_bands_3d(e, size=75, angle=30):
    """
    Plots each bands and their average in three 3D plots.
    """
    x, y = np.meshgrid(range(size), range(size))
    b1 = e.band_1.reshape(size, size)[(x, y)]
    b2 = e.band_2.reshape(size, size)[(x, y)]

    f = plt.figure()

    ax = f.add_subplot(131, projection='3d')
    ax.set_title('band 1')
    ax.plot_surface(x, y, b1, cmap=cm.coolwarm)
    ax.view_init(30, angle)

    ax = f.add_subplot(132, projection='3d')
    ax.set_title('band 2')
    ax.plot_surface(x, y, b2, cmap=cm.coolwarm)
    ax.view_init(30, angle)
    
    ax = f.add_subplot(133, projection='3d')
    ax.set_title('band average')
    ax.plot_surface(x, y, (b1 + b2) / 2, cmap=cm.coolwarm)
    ax.view_init(30, angle)
    
    
def plot_graph_steps(graphs):
    """
    Plots successively the adjency matrix and the networkx spring representation.
    """
    for g in graphs:
        if sp.sparse.issparse(g):
            g = g.todense()
        
        plt.subplot(121)
        plt.title('adjency matrix')
        plt.spy(g)
        plt.subplot(122)
        nx.draw(nx.from_numpy_array(g))
        plt.show()
