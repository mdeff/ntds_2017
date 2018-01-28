from sklearn.cluster import SpectralClustering
from gcnn.coarsening.coarse_utils import *

def coarsen(A, levels, self_connections=False):
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple
    levels.
    """
    graphs, parents = amg(A, levels)
    perms = compute_perm_amg(parents)

    for i, A in enumerate(graphs):
        M, M = A.shape

        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

    return graphs, perms[0] if levels > 0 else None

def amg(W, levels=2, rid=None):
    """
    Coarsen a graph multiple times using the algebraic multigrid algorithm.
    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs
    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 = N_1/2
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    for _ in range(levels):

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        weights = degree            # graclus weights
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = sp.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        
        # COMPUTE ONE LEVEL OF AMG CLUSTERING
        cluster = SpectralClustering(n_clusters=int(N/2.), eigen_solver='amg', affinity='precomputed', n_jobs=-1, n_init=1)
        cluster_id = cluster.fit_predict(W)
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = sp.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents