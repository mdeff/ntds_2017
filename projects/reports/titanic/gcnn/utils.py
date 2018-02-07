from skorch import NeuralNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from timeit import default_timer as timer
import numpy as np 
from skorch.utils import to_numpy
from sklearn.metrics import log_loss


def score_classification(truth, predicted):
    """
    Score according to accuracy, precision, recall and f1.
    """
    print(classification_report(truth, predicted, target_names=['ship', 'iceberg']))
    return [
        accuracy_score(truth, predicted),
        precision_score(truth, predicted),
        recall_score(truth, predicted),
        f1_score(truth, predicted)
    ]


class NNplusplus(NeuralNet):
    '''
    inherit NeuralNet class from skorch
    '''
    
    def score(self,X,target):
        '''
        redefine scoring method to be the same as the one of kaggle (log_loss)
        '''
        y_preds = []
        for yp in self.forward_iter(X, training=False):
            y_preds.append(to_numpy(yp.sigmoid()))   
        y_preds = np.concatenate(y_preds, 0)
        return log_loss(target,y_preds)
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

