# -*- coding: utf-8 -*-

""" 
    Install customized surprise library from GitHub 
    https://github.com/manzo94/Surprise/tree/laplacian_smooth
    to run the code
    
    This files contains some template to run cross validation
    and grid search with the surprise library. Select the algorithm
    you want to test and call one of the functions on them
"""

import Dataset as ds
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy, SVDsmooth
from surprise.dataset import DatasetUserFolds
from surprise.model_selection import GridSearchCV, cross_validate, split
import pygsp as gsp
import matplotlib.pyplot as plt

def global_mean():
    """ Global Mean prediction """
    trainset, testset = split.train_test_split(data, test_size=.17, random_state=1)
    labels = list(zip(*testset))[2]
    err = labels - trainset.global_mean
    rmse = np.sqrt(np.sum(err**2) / len(testset))
    print('RMSE with global mean: ',rmse)

def score_on_predefined_trainset(algo, dd):
    """ Code to evaluate the score of an algorithm over a predefined trainset
        and testset which are contained in dd. This can be useful to make
        comparison with other algorithms from other packages. 
    """
    # Import predefined trainset and testset
    try:
        data = DatasetUserFolds(reader=Reader())
    except:
        # We are forcing this class to build without some necessary parameter
        # so we need to skip the errors raised
        pass
    
    raw_trainset = [(uid, iid, r*4+1, None)
                                    for (uid, iid, r) in
                                    dd.train.itertuples(index=False)]
    raw_testset = [(uid, iid, r*4+1, None)
                                    for (uid, iid, r) in
                                    dd.test.itertuples(index=False)]
    trainset = data.construct_trainset(raw_trainset)
    testset = data.construct_testset(raw_testset)

    algo.fit(trainset)
    predictions = algo.test(testset)
    return accuracy.rmse(predictions)

def cross_validation(data, algo):
    """ Cross Validation Template """
    cross_validate(algo, data, measures=['rmse'], cv=6, n_jobs=1, verbose=True)

def grid_search():
    """ grid search template """
    
    # Set Grid Parameters
    G = gsp.graphs.Graph(dd.build_friend_friend())
    G.compute_laplacian('normalized')
    param_grid = {
            #'L' : [G.L.todense()],
            'n_factors' : [5],
            'n_epochs' : [30],
            'lr_all' : [1.e-3],
            'reg_all' : np.logspace(-6,-1, 20),
            #'reg' : np.logspace(-6,-1,15)
    }
    
    # Init grid_search
    grid = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=6, n_jobs=1, joblib_verbose=10000)
    grid.fit(data)
    
    # Print best score and best parameters
    print('Best Score: ', grid.best_score['rmse'])
    print('Best parameters: ', grid.best_params['rmse'])
    
    # Plot RMSE
    plt.plot(grid.cv_results['param_reg_all'], grid.cv_results['mean_test_rmse'])

dd = ds.Dataset()
dd.prune_ratings()
dd.prune_friends()
dd.normalize_weights()
dd.split(test_ratio=0.17, seed=1)

data = Dataset(Reader())
# If you want to use artist network for smoothing, swap here uid with iid
data.raw_ratings = [(uid, iid, r*4+1, None)
                                for (uid, iid, r) in
                                dd.ratings.itertuples(index=False)]

# Use pygsp to build Laplacian, choose between artist or user network
G = gsp.graphs.Graph(dd.build_friend_friend())
#G = gsp.graphs.Graph(dd.build_art_art())
G.compute_laplacian('normalized')

# Choose between the two algorithms
algo = SVD(n_factors=5, n_epochs=20, lr_all=1.-3, reg_all=1.-4)
algo = SVDsmooth(G.L.todense(), n_factors=5, n_epochs=20, lr_all=1.-3, alpha=1.-4)

