#coding: utf-8
from abc import ABC, abstractmethod
from scipy import sparse
import scipy.sparse.linalg
import numpy as np


class Deformation(ABC):
    """
    Abstraction for meshs deformation
    """
    @abstractmethod
    def transform(self, src, parameters):
        """
        Apply a given transformation to the input

        :param src:         Object to apply deformation on
        :param parameters:  Transformation parameters
        :return:            Transformed object
        """
        pass

    @abstractmethod
    def jacobian(self, src, parameters):
        """
        Compute the deformation jacobian for a given instance (object parameter)

        :param src:         Current object
        :param parameters:  Current parameters
        :return:            Jacobian
        """
        pass

def deform_anchor(source, target, sel, lap, alpha):
    """
    Solve the system argmin | sel ( src + d) - tgt |

    :param source:      Surface to deform
    :param target:      Surface to reach
    :param sel:         Selected anchor
    :param lap:         Laplacian operator
    :param alpha:       Regularizer
    :return:            Estimated surface and deformation field
    """
    # Parameters
    xs = source
    xt = target
    d = np.zeros(xs.shape, dtype=np.float32)
    # Define linear system
    ef = sel.dot(xs) - xt
    A = sel.transpose().dot(sel)
    b = -(sel.transpose().dot(ef))
    # Solve
    for k in range(3):
        d[:, k] = sparse.linalg.lsqr(A, b[:, k])[0]
    # Reconstruct target
    estm_tgt = xs + d
    return estm_tgt, d


def deform_regularized_anchor(source, target, sel, lap, alpha):
    """
    Solve the system argmin | sel ( src + d) - tgt | + d'Ld

    :param model:       Deformation model
    :param source:      Surface to deform
    :param target:      Surface to reach
    :param sel:         Selected anchor
    :param lap:         Laplacian operator
    :param alpha:       Regularizer
    :return:            Estimated surface and deformation field
    """
    # Parameters
    xs = source
    xt = target
    d = np.zeros(xs.shape, dtype=np.float32)
    # Define linear system to solve
    ef = sel.dot(xs) - xt
    A = sel.transpose().dot(sel) + alpha * lap
    b = -(sel.transpose().dot(ef))
    # Solve
    for k in range(3):
        d[:, k] = sparse.linalg.lsqr(A, b[:, k])[0]
    # Estimation
    estm_tgt = xs + d
    # Done
    return estm_tgt, d


def deform_mesh(mesh, anchors, idx, weight = 1.0):
    """
    Deform a given mesh in order to match some target points (annchors)

    :param mesh:    Mesh to deform (object, vertex [N x 3])
    :param anchors: Target points, position to reach [K x 3]
    :param idx:     Index of the corresponding anchors
    :param weights: Anchor's weight
    :return:        Deformed surface estimated
    """
    # Define dimensions
    N = mesh.vertex.shape[0]
    K = anchors.shape[0]
    estm_xt = np.zeros(mesh.vertex.shape, dtype=mesh.vertex.dtype)

    # Compute laplacian + augment it with anchor's index
    _, _, lap = mesh.compute_laplacian('cotan')
    ridx = lap.row.tolist()
    cidx = lap.col.tolist()
    data = lap.data.tolist()
    for k in range(K):
        ridx.append(N + k)
        cidx.append(idx[k])
        data.append(weight)
    lap = sparse.csr_matrix((data, (ridx, cidx)), shape=(N+K, N), dtype=lap.dtype)
    # Compute deltas + augment with anchors
    deltas = lap.dot(mesh.vertex)
    for k in range(K):
        deltas[N + k, :] = anchors[k,:] * weight
    # Update each dimension
    for k in range(3):
        estm_xt[:, k] = scipy.sparse.linalg.lsqr(lap, deltas[:, k])[0]
    # Done
    return estm_xt

"""
# --------------------------------------------------------
# Deformation model
# --------------------------------------------------------
class TranslationDeformation(Deformation):
    
    
    

    def __init__(self, n):
    
        self._n = n

    def transform(self, src, parameters):
        return src + parameters

    def jacobian(self, src, parameters):
        return np.eye(self._n, self._n, dtype=np.float32)


class AffineDeformation(Deformation):

    def __init__(self, n):
        self._n = n

    def transform(self, src, parameters):
        qs = parameters[0:self._n * 3].reshape((-1, 3))
        ts = parameters[self._n * 3:].reshape((-1, 3))
        xs = src.reshape((-1, 3))
        xt = np.zeros(src.shape, dtype=src.dtype).reshape((-1, 3))
        for k in range(self._n):
            rmat = self._rot_matrix([1.0, qs[k, 0], qs[k, 1], qs[k, 2]])
            xt[k] = rmat @ xs[k].T + ts[k].T
        return xt.reshape((-1, 1))

    def jacobian(self, src, parameters):
        N = int(src.shape[0] / 3)
        J = np.zeros((N * 3, N * 6))
        xs = src.reshape((-1, 3))
        x = xs[:, 0]
        y = xs[:, 1]
        z = xs[:, 2]
        q = parameters[0:3*N].reshape(-1, 3)
        #qn = np.linalg.norm(q, axis=1)
        q1 = q[:, 0]
        q2 = q[:, 1]
        q3 = q[:, 2]
        q0 = np.sqrt(1.0 - ((q1 * q1) + (q2 * q2) + (q3 * q3)))
        # Jt
        # Translation
        J[:, 3 * N:] = np.eye(3*N, 3*N)
        # Rotation
        idxx = range(0, 3*N, 3)
        idxy = range(1, 3*N, 3)
        idxz = range(2, 3*N, 3)
        # Q0 ---------
        #idxq = range(0, 4 * N, 4)
        # dXdQ0
        #J[idxx, idxq] = 2.0 * ((-q3 * y) + (q2 * z))
        # dYdQ0
        #J[idxy, idxq] = 2.0 * ((q3 * x) - (2.0 * (q0 * y)) - (q1 * z))
        # dZdQ0
        #J[idxz, idxq] = 2.0 * ((-q2 * x) + (q1 * y))
        # Q1 ---------
        idxq = range(0, 3 * N, 3)
        # dXdQ1
        J[idxx, idxq] = 2.0 * ((q2 * y) + (q3 * z))
        # dYdQ1
        J[idxy, idxq] = 2.0 * ((q2 * x) - 2.0 * (q1 * y) - (q0 * z))
        # dZdQ1
        J[idxz, idxq] = 2.0 * ((q3 * x) - 2.0 * (q1 * z) + (q0 * y))
        # Q2 ---------
        idxq = range(1, 3 * N, 3)
        # dXdQ2
        J[idxx, idxq] = 2.0 * (-2.0 * (q2 * x) + (q1 * y) + (q2 * z))
        # dYdQ2
        J[idxy, idxq] = 2.0 * ((q1 * x) + (q3 * z))
        # dZdQ2
        J[idxz, idxq] = 2.0 * (-(q0 * x) + (q3 * y) - 2.0 * (q2 * z))
        # Q3 ---------
        idxq = range(2, 3 * N, 3)
        # dXdQ3
        J[idxx, idxq] = 2.0 * (-2.0 * (q3 * x) - (q0 * y) + (q1 * z))
        # dYdQ3
        J[idxy, idxq] = 2.0 * ((q0 * x) + (q2 * z))
        # dZdQ3
        J[idxz, idxq] = 2.0 * ((q1 * x) + (q2 * y))
        # Done
        return J

    def _rot_matrix(self, q):
        qn = q / np.linalg.norm(q)
        rmat = np.zeros((3, 3), dtype=q.dtype)
        rmat[0, 0] = 0

        return rmat


shift_model = TranslationDeformation(surf.size)
affine_model = AffineDeformation(surf.shape[0])

"""