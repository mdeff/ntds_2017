from scipy.sparse.linalg import eigsh
from scipy.sparse import linalg
from scipy import sparse, stats
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import numpy as np

def calc_weight(n, root, pred, local_tree):
    
    if n==root:
        return 0, root, root
    
    parent = pred[root, n]

    w_p = local_tree[parent, n]
    
    gparent = pred[root, parent]
    
    if gparent != -9999:
        w_d = local_tree[gparent, parent]
    else:
        w_d = 1
        gparent = n
        
    w = 2./(1./w_p + 1./w_d)

    return w, parent, gparent

def mst(graph, levels=2):
    
    G = [graph]
    
    for _ in range(levels):
        graph, _ = one_level_MST(graph)
        G.append(graph)
        
    return G

def one_level_MST(test_dist):

    test_dist_triu = np.triu(test_dist)
    Tree = minimum_spanning_tree(test_dist_triu)
    Tree = Tree + Tree.T
    local_tree = Tree.todense()

    distance_matrix, pred = dijkstra(Tree.todense(), directed=False, unweighted=True, return_predecessors=True)

    root = np.random.choice(np.arange(distance_matrix.shape[0]))
    even_nodes = distance_matrix[:, root] % 2 == 0
    even_nodes = np.arange(test_dist.shape[0])[even_nodes]

    weight_tree = np.zeros((int(local_tree.shape[0]/2), int(local_tree.shape[1]/2)))
    
    array = [-1]*local_tree.shape[0]
    cluster = 0
    
    for n in even_nodes:
        new_weight, parent, gparent = calc_weight(n, root, pred, local_tree)
        array[n] = cluster
        array[parent] = cluster
        cluster += 1
        
        n = int(n/2)
        out_div = int(gparent/2)
        weight_tree[n, out_div] = new_weight
        weight_tree[out_div, n] = new_weight
        
    return weight_tree, array