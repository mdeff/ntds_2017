import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, spatial


def features_to_dist_matrix(features, metric="cosine"):
    """
    Compute the (square) distance matrix given some features and a metric

    Parameters
    ----------
    features : ndarray
    features matrix

    metric : string
    The distance metric to use. The distance function can
    be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'matching',
    'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.


    Returns
    -------
    out : ndarray
    Output array, distance matrix
    """

    return spatial.distance.squareform(spatial.distance.pdist(features, metric=metric))


def dist_to_adj_matrix(dist, kernel_type):
    """
    Given a distance matrix, build an adjacency matrix using a given kernel_type

    Parameters
    ----------
    dist : ndarray
    distance matrix

    kernel_type : string
    kernel type ('gaussian' or 'linear')


    Returns
    -------
    out : ndarray
    Output array, adjacency matrix
    """

    if kernel_type == 'gaussian':
        kernel_width = dist.mean()
        mat = np.exp(-(dist / kernel_width) ** 2)

    elif kernel_type == 'linear':
        mat = 1 - dist / dist.max()

    else:
        raise ValueError('kernel_type unknown')

    np.fill_diagonal(mat, 0.0)
    return mat


def filter_neighbors(weights, neighbors):
    """
    Given a weights matrix, keep only the best edges/neighbors (edges with biggest weight)
    for each node

    Parameters
    ----------
    mat : ndarray
    weights matrix

    neighbors : int
    number of best edges/neighbors (at least) to keep for each node

    Returns
    -------
    out : ndarray
    Output array, the same weights matrix but with only the best edges/neighbors
    """
    ind = np.argsort(weights, axis=1)[:, ::-1][:, :neighbors]
    new_weights = np.zeros(weights.shape)
    for idx, indices in enumerate(ind):
        new_weights[idx, indices] = 1

    new_weights = ((new_weights.T + new_weights) != 0) * weights
    return new_weights


def plot_labels(eigenvectors, labels=None, ax=plt):
    """
    Scatter plot the labels onto value of the second eigenvector as the x coordinate of a node,
    and the value of the third eigenvector as the y coordinate

    Parameters
    ----------
    eigenvectors : ndarray
    eigenvectors used for the axis and (Fiedler vector) for the labels if none are given

    labels : ndarray
    labels to plot

    ax : matplotlib axis
    axis on which the plot is plotted

    Returns
    -------
    None
    """

    ax.scatter(eigenvectors[:, 1], eigenvectors[:, 2], c=labels, cmap='Accent', alpha=0.5)


def compute_err(labels, target_labels, print_=False):
    """
    Compute the error rate between labels (prediction) and genres (truth)

    Parameters
    ----------
    labels : ndarray
    predicted labels

    genres : ndarray
    genres (ground truth labels)

    print_ : boolean
    whether or not to print the number of errors and error rate

    Returns
    -------
    out : float
    Output error rate
    """
    err = np.abs(target_labels - labels).sum()
    perc_err = err / len(labels)

    # little hack :
    if perc_err > 0.5:
        # this is to cope with the fact that sometimes labels given by the eigenvectors
        # are the inverse of the real genres ('> 0' instead of '< 0')
        err = len(labels) - err
        perc_err = 1 - perc_err
    if print_:
        print('{} errors ({:.2%})'.format(err, perc_err))
    return perc_err


def compare_plot_labels(eigenvectors, labels):
    """
    Scatter plot the labels (both the ones given by the Fiedler vector and the ground truth (genres))
    onto value of the second eigenvector as the x coordinate of a node,
    and the value of the third eigenvector as the y coordinate

    Parameters
    ----------
    eigenvectors : ndarray
    eigenvectors used for the axis and for the labels if none are given

    genres : ndarray
    genres (ground truth labels)

    Returns
    -------
    None
    """
    f, axes = plt.subplots(nrows=1, ncols=2)
    spectral_labels = eigenvectors[:, 1] > 0
    for ax, labels, title in zip(axes.flatten(), [spectral_labels, labels],
                                 ["spectral embedding", "pop vs rock clutstering"]):
        ax.set_title(title)
        plot_labels(eigenvectors, labels, ax=ax)

    compute_err(spectral_labels, target_labels=labels, print_=True)


def fast_spectral_decomposition(X, vectorizer=None, new_texts=None, metric='cosine', kernel_type='gaussian',
                                neighbors=100, n_eigen=5, return_eigenvalues=False):
    if vectorizer is not None:
        assert new_texts is not None
        new_x_sparse = vectorizer.transform(new_texts)
        new_x = sparse.csr_matrix.todense(new_x_sparse)
        X = np.append(X, new_x, axis=0)

    distances = features_to_dist_matrix(X, metric)
    all_weights = dist_to_adj_matrix(distances, kernel_type)
    weights = sparse.csr_matrix(filter_neighbors(all_weights, neighbors))
    laplacian = sparse.csgraph.laplacian(weights, normed=True)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian, k=n_eigen, which='SM')

    if return_eigenvalues:
        return eigenvalues, eigenvectors

    return eigenvectors
