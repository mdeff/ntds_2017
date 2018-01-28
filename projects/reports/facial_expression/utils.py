import struct
import os
import re
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm, tnrange, tqdm_notebook
import scipy
import matplotlib.pyplot as plt
from scipy import sparse, stats, spatial
import scipy.sparse.linalg


def prepare_data(src, dest):
    """
    Extract FaceWarehouse meshes from dataset folder (src) and copy them into
     dest folder
    :param src:     Location of FaceWarehouse meshes
    :param dest:    Location where to export them
    """
    meshes = []
    srch = re.compile(r"_([0-9]*)/")
    # Scan folders
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.endswith('bs'):
                # Extract subject ID
                fname = os.path.join(root, f)
                match = srch.search(fname)
                if match:
                    sid = int(match.groups()[0]) - 1
                    meshes.append((sid, fname[len(src):]))
                else:
                    print('Error, can not determine subject ID')
    # Copy files
    if not os.path.exists(dest):
        os.makedirs(dest)
    for m in meshes:
        sid = m[0]
        path = m[1]
        fname = path.split('/')[-1]
        fdest = dest + '%03d_' % sid + fname
        if not os.path.isfile(fdest):
            print('Copy mesh: %s' % path)
            shutil.copy2(src + path, fdest)
    print('Done')


def load_triangulation(filename):
    """
    Load triangulation file
    :param filename:    Path to triangulation file (*.tri)
    :return:            Numpy array [N x 3] holding vertices index forming
                        each triangle
    """
    tris = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            tris.append(int(parts[1]) - 1)
            tris.append(int(parts[2]) - 1)
            tris.append(int(parts[3]) - 1)
    if len(tris) != 0:
        return np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    else:
        print('Error, no triangle loaded')


def load_meshes(filename, exps=None):
    """
    Load meshes from binary files
    :param filename:    Path to binary mesh file
    :param exps:        List of expression to load (indices, or None)
    :return:            List of of numpy array holding meshes K * [N x 3]
    """
    meshes = []
    with open(filename, 'rb') as f:
        # Read header
        nexp = struct.unpack('@i', f.read(4))[0]
        nvert = struct.unpack('@i', f.read(4))[0]
        #  x,y,z components
        nvert *= 3
        npoly = struct.unpack('@i', f.read(4))[0]
        # Select expression
        idx = range(0, nexp + 1) if exps is None else sorted(list(set(exps)))
        # Iterate over selection
        for i in idx:
            # Move to proper section
            offset = i * nvert * 4 + 12
            f.seek(offset, 0)
            # Read vertices
            vertex = struct.unpack('@{}f'.format(nvert), f.read(nvert * 4))
            meshes.append(np.asarray(vertex, dtype=np.float32).reshape(-1, 3))
    if len(meshes) == 0:
        print('Error, can not load meshes from file: %s' % filename)
    return meshes


def save_obj(vertex, tri, filename):
    """
    Dump a mesh into an obj file
    :param vertex:      Matrix with vertex
    :param tri:         Matrix with triangle
    :param filename:    Path where to save the *.obj
    """
    with open(filename, 'w') as f:
        for v in vertex:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for t in tri:
            f.write('f %d %d %d\n' % (t[0] + 1, t[1] + 1, t[2] + 1))


def load_obj(filename):
    """
    Load mesh from an *.obj file

    :param filename:    Path to obj file
    :return:            Tuple with a vertex, tri
    """
    vertex = []
    tri = []
    with open(filename, 'r') as f:
        for line in f:
            # Vertex ?
            if line[0:2] == 'v ':
                p = line.strip().split(' ')
                vertex.append(float(p[1]))
                vertex.append(float(p[2]))
                vertex.append(float(p[3]))
            # Triangle ?
            elif line[0:2] == 'f ':
                p = line.strip().split(' ')
                assert len(p) == 4
                tri.append(int(p[1]) - 1)
                tri.append(int(p[2]) - 1)
                tri.append(int(p[3]) - 1)

    return np.asarray(vertex, dtype=np.float32).reshape((-1, 3)), \
           np.asarray(tri, dtype=np.int32).reshape((-1, 3))


def gather_neighbour(tri):
    """
    Generate the list of neighboring vertex indexes from a triangle list
    :param tri: Triangulation matrix
    :return:    List of neighbour (list(set))
    """
    #  number of element polygon
    spoly = tri.shape[1]
    assert (spoly == 3)
    # Vertex max
    N = tri.max()
    # Define neighbor
    neighbour = [set() for _ in range(N + 1)]
    for l in tri:
        for k in range(spoly):
            v0 = l[k]
            e1 = l[(k + 1) % spoly]
            e2 = l[(k + 2) % spoly]
            neighbour[v0].add(int(e1))
            neighbour[v0].add(int(e2))
    return list(map(list, neighbour))


def load_anchor_point(filename):
    """
    Load selected anchors points from file

    :param filename:    Path to anchor's file
    :return:            List of anchor's index
    """
    idx = []
    with open(filename, 'r') as f:
        for line in f:
            idx.append(int(line.strip()))
    return idx


def load_set(ratio_train, folder_path):
    """
    Load dataset and split it into train set and test set given the ratio given.

    :param ratio_train:    ratio of the dataset dedicated to the training set
    :param folder_path:    Location of the dataset
    """
    filenames = pd.Series(os.listdir(folder_path)).sort_values()

    len_train_set = int(round(ratio_train * len(filenames), 0))
    len_test_set = len(filenames) - len_train_set

    individuals = []

    for fname in tqdm_notebook(filenames[:len_train_set],
                               desc='Load train Set'):
        faces = load_meshes(folder_path + fname)
        individuals.append(faces)

    individuals_test = []

    for fname in tqdm_notebook(filenames[len_train_set:], desc='Load Test Set'):
        faces = load_meshes(folder_path + fname)
        individuals_test.append(faces)

    return (individuals, individuals_test)


def construct_features_matrix(individuals, nb_face):
    """
    Load dataset and split it into train set and test set given the ratio given.

    :param individuals:    set of all the faces for a given set of individuals (ex: train_set)
    :param nb_face:   Number of faces per person we wish to analyse
    """

    expression_label = [j for i in range(0, len(individuals)) for j in
                        range(1, nb_face)]

    features = []

    if nb_face:
        for i, ind in enumerate(
                tqdm_notebook((individuals), desc='Built Features')):
            for j, face in enumerate(ind[1:nb_face]):
                if (i == 0 and j == 0):
                    features = face.reshape(1, -1) - ind[0].reshape(1, -1)
                else:
                    err = face.reshape(1, -1) - ind[0].reshape(1, -1)
                    features = np.vstack((features, err))
    else:
        for i, ind in enumerate(
                tqdm_notebook((individuals), desc='Built Features')):
            if i == 0:
                features = ind[0].reshape(1, -1)
            else:
                new = ind[0].reshape(1, -1)
                features = np.vstack((features, new))

    return expression_label, features


def compute_laplacian(features, nb_neighbours, distance_metric, nb_eigen,
                      verbose):
    """
    Compute distances between each sample, compute weight matrix from the distances, Laplacian and return eigenvalues/vectors

    :param features:    ndarray, set of all the faces for a given set of individuals (ex: train_set)
    :param nb_neighbours:   int, Number of neighbours per sample we want to keep in order to sparcify the weight matrix
    :param distance_metric:   str, method to compute distance between samples (ex : 'Euclidean', 'Cosine')
    :param nb_eigen:   int, Number of first eigenvalues/vectors we want to compute
    :param verbose:   int, 1 to display graphs, 0 to display nothing
    """

    # Compute distances between every samples
    print('Compute Distances')
    distances = scipy.spatial.distance.pdist(features, metric=distance_metric)
    distances = scipy.spatial.distance.squareform(distances)

    if verbose:
        plt.hist(distances.reshape(-1), bins=50)
        plt.xlabel('Distance')
        plt.ylabel('Number')
        plt.title('Distance distribution')
        plt.show()

    # Compute weights from distance matrix
    kernel_width = distances.mean()
    weights = np.exp((-distances ** 2) / (kernel_width ** 2))
    np.fill_diagonal(weights, 0)

    # Sparsen weight matrix with nb_neighbours
    sparse_weights = np.zeros((len(weights), len(weights)), dtype=np.float)

    for i in tqdm_notebook(range(len(sparse_weights)), desc='Sparse Weight'):
        ord_index = np.argsort(weights[i, :])
        for j in ord_index[-nb_neighbours:]:
            sparse_weights[i, j] = weights[i, j]

            if not sparse_weights[j, i]:  # Force symmetry in the matrix
                sparse_weights[j, i] = weights[i, j]

    # Compute laplacian from the sparse weight matrix
    print('Compute Laplacian')
    laplacian = sparse.csgraph.laplacian(sparse_weights, normed=True)
    laplacian = sparse.csr_matrix(laplacian)

    # Compute eigenvalues/vectors from laplacian
    print('Compute Eigenvalues/vectors')
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian, k=nb_eigen,
                                                          which='SM')

    if verbose:
        plt.plot(eigenvalues, '.-', markersize=15);
        plt.xlabel('Eigenvalues')
        plt.ylabel('Values')
        plt.title(str(nb_eigen) + ' first eigenvalues')
        plt.show()

    return eigenvalues, eigenvectors


class MeshGenerator:
    """
    Utility class to generate toy examples
    """
    def make_plane(self, nx, ny, dx=1.0, dy=1.0):
        """
        Generate a simple plane aligned with the XY plane

        :param nx:  Number of step in the X direction
        :param ny:  Number of stop in the Y direction
        :param dx:  Increment size in X
        :param dy:  Increment size in Y
        :return:    Surface, Triangulation
        """
        nvertex = nx * ny
        pts = np.zeros((nvertex, 3), dtype=np.float32)
        tri = []
        for ky in range(0, ny):
            for kx in range(0, nx):
                # Define position
                x = kx * dx
                y = ky * dy
                z = 0.0
                idx = kx + nx * ky
                pts[idx, :] = [x, y, z]
                # Define triangle
                if ky > 0 and kx > 0:
                    v2 = idx
                    v1 = idx - 1
                    v3 = idx - nx
                    v0 = v3 -1
                    tri.append([v0, v1, v3])
                    tri.append([v1, v2, v3])
        return pts, np.asarray(tri, dtype=np.int32).reshape((len(tri), 3))

    def transform(self, points, func):
        """
        Apply a given transform to a set of points [Nx3]

        :param points:  Point set on which to apply the transformation
        :param func:    Deformation to apply on the points
        :return:
        """
        pts = np.zeros(points.shape, points.dtype)
        for i, r in enumerate(points):
            pts[i, :] = func(r)
        return pts
