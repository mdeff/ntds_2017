import numpy as np
from tqdm import tqdm_notebook as tqdm

import spectral


def grid_search(X, param_grid):
    """
    Compute all error rates for the given combinations of parameters

    Parameters
    ----------
    param_grid : sklearn model_selection ParameterGrid
    grid of parameters (all combinations to try)

    Returns
    -------
    out : parameters, ndarray
    Output the best parameters and all the error rates
    """
    errors = [np.sum(spectral.fast_spectral_decomposition(X, return_eigenvalues=True, n_eigen=4, **p)[0])
              for p in tqdm(param_grid)]
    return param_grid[np.argmin(errors)], errors
