import numpy as np
from sklearn import linear_model


def crossvad_build_k_indices(y, k_fold, seed = 1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k_fold):
    """use cross validation on method. 
        return the mean of the mse error of the training sets and of the testing sets
        as well as the accuracy and the variance (over kfold) of the accuracy"""
    # ***************************************************
    #creating list of possible k's
    k_list=np.arange(k_indices.shape[0])   
 
    # define lists to store the w, the loss of training data and the loss of test data
    mse_tr_list = np.zeros(k_fold)
    mse_te_list = np.zeros(k_fold)
    accuracy_te_list = np.zeros(k_fold) 
    y_pr_stack = np.zeros(len(y))
    for k in range(0,k_fold):
        # get k'th subgroup in test, others in train  
        y_te = y[k_indices[k]]
        x_te = x[k_indices[k]]
        y_tr = y[np.ravel(k_indices[k_list[k*np.ones(len(k_list))!=k_list]])]
        x_tr = x[np.ravel(k_indices[k_list[k*np.ones(len(k_list))!=k_list]])]
        #standardize the data
        x_tr, mean_tr, std_tr = standardize(x_tr)
        x_te = standardize_given(x_te, mean_tr, std_tr)
        
        #logistic regression
        logreg = linear_model.LogisticRegression(solver ='liblinear', class_weight ='balanced')
        logreg.fit(x_tr, y_tr)
        y_pr = logreg.predict(x_te)

        y_pr_stack[int(k*len(y)/k_fold):int((k+1)*len(y)/k_fold)] = y_pr
        accuracy_te_list[k] = sum(np.equal(y_pr,y_te))/len(y_te)
        
    mse_tr_mean = np.mean(mse_tr_list)
    mse_te_mean = np.mean(mse_te_list)
    accuracy_te_mean = np.mean(accuracy_te_list)
    accuracy_te_var = np.std(accuracy_te_list)
    return y_pr_stack, accuracy_te_mean, accuracy_te_var

def standardize(x):
    """Standardize the original data set."""
    #standardize is done feature by feature to have equal weights. 
    mean_x = np.mean(x,axis=0)
    x = x - mean_x
    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x

def standardize_given(x, mean_x, std_x):
    """Standardize the original data set with given mean_x and std_x."""
    x = x - mean_x
    x = x / std_x  #handle outliers
    return x