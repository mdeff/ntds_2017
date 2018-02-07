import itertools

import numpy as np
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture


def find_best_perm(y_true, y_pred, metric=lambda y1, y2: f1_score(y1, y2, average='weighted')):
    """
    Find the best permutations of all possible cluster assignment

    Parameters
    ----------
    y_true : ndarray
    true labels

    y_pred : ndarray
    prediction labels

    metric : numpy function
    metric function to optimize (higher is better) (default is f1_score)

    Returns
    --------
    out : tuple
    Output tuple of the permutations
    """
    scores = []
    permutations = list(itertools.permutations(range(len(set(y_true)))))
    for perm in permutations:
        apply_perm = np.vectorize(lambda x: perm[x])
        score = metric(y_true, apply_perm(y_pred))
        scores.append(score)

    return permutations[np.argmax(scores)]


def fast_gmm(y_true, n_classes, eigenvectors):
    gmm_clf = GaussianMixture(n_components=n_classes, covariance_type='full', max_iter=500, random_state=42)
    gmm_clf.fit(eigenvectors)

    y_pred_brute = gmm_clf.predict(eigenvectors)
    y_pred_proba_brute = gmm_clf.predict_proba(eigenvectors)

    best_perm = find_best_perm(y_true, y_pred_brute)

    y_pred = np.vectorize(lambda x: best_perm[x])(y_pred_brute)
    d = {best_perm[i]:i for i in range(len(best_perm))}
    perm = np.array([ d[i] for i in range(len(d))])
    y_pred_proba = y_pred_proba_brute[:, perm]
    return y_pred, y_pred_proba
