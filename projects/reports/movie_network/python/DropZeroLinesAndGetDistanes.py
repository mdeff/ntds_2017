import sys, os, copy
import numpy as np
import pandas as pd

from sklearn import preprocessing, decomposition
from scipy import sparse, stats, spatial
import scipy.sparse.linalg

def DropZeroLinesAndGetDistanes(Frames):
    AllIndex = Frames[0].index.values
    CleanedFrames=[]
    CleanedFrames.append( Frames[0][(Frames[0].T != 0).any()] )
    CleanedFrames.append( Frames[1][(Frames[1].T != 0).any()] )
    CleanedFrames.append( Frames[2][(Frames[2].T != 0).any()] )

    DropLines0 = (list(set(AllIndex) - set(CleanedFrames[0].index.values)))
    DropLines1 = (list(set(AllIndex) - set(CleanedFrames[1].index.values)))
    DropLines2 = (list(set(AllIndex) - set(CleanedFrames[2].index.values)))

    DropLines = set(DropLines0 + DropLines1 + DropLines2)
    Remainers = list(set(AllIndex) - set(DropLines))
    print("Remaining % of movies:",len(Remainers)/len(Frames[0]))

    newFrames=[]
    for i in range(len(Frames)):
        newFrames.append(Frames[i].drop(list(DropLines)))
    
    all_distances=[]
    for Frame in newFrames:
        all_distances.append(np.nan_to_num(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Frame, metric='cosine'))))   
    
    return all_distances, list(DropLines), newFrames