import sys, os, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PlotMatrix(Matrix):
    plt.rcParams['figure.figsize'] = (17, 9)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.rcParams['figure.figsize'] = (17, 9)
    plt.imshow(Matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()