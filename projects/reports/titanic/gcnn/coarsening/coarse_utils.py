import scipy as sp
import numpy as np

def compute_perm_amg(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    return indices[::-1]

def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    if Mnew > M:
        rows = sp.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = sp.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = sp.sparse.vstack([A, rows])
        A = sp.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is sp.sparse.coo.coo_matrix
    return A