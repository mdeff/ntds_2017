
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Math
import numpy as np
import scipy.stats
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
from scipy import sparse, stats, spatial
import scipy.sparse.linalg

# Machine learning
from sklearn.utils import shuffle
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd

# Cutting

from cut_audio import *




def main_train_audio_extraction():
    '''
    - Function that allow the extraction of all the audio files.
    - Process :
    1. Indexing the path, the class and the speaker of all the audio files.
    2. Audio Extraction :
        2.1. Loading all the audio files into memory
        2.2. Detecting the position of the word inside each audio files and cutting them
        2.3. Saving into a Pickled DataFrame all the audio and their cutted version
    '''

    train_audio_path = join('..','Project','data','train','audio')

    # Listing the directories of each word class
    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()

    path = []
    word = []
    speaker = []
    iteration = []

    # Loading the information of the audio files
    for direct in dirs:
        if not direct.startswith('_'):

            list_files = os.listdir(join(train_audio_path, direct))
            wave_selected  = list([ f for f in list_files if f.endswith('.wav')])

            # Extraction of file informations for dataframe
            word.extend(list(np.repeat(direct,len(wave_selected),axis=0)))
            speaker.extend([wave_selected[f].split('.')[0].split('_')[0] for f in range(len(wave_selected)) ])
            iteration.extend([wave_selected[f].split('.')[0].split('_')[-1] for f in range(len(wave_selected)) ])
            path.extend([train_audio_path + '/' + direct + '/' + wave_selected[f] for f in range(len(wave_selected))])

    # Saving those informations into a pandas DataFrame
    features_og = pd.DataFrame({('info','word',''): word,
                                ('info','speaker',''): speaker,
                                ('info','iteration',''): iteration,
                                ('info','path',''): path})
    index_og = [('info','word',''),('info','speaker',''),('info','iteration','')]

    print('Number of signals : ' + str(len(features_og)))


    # Load and cut the audio files.
    raw_audio_df = load_audio_file(features_og)

    # Save the raw audio Dataframe into a set a pickles :
    i = 0
    k = 0
    while True :
        i_next = i + 6000
        k += 1
        if i_next < len(raw_audio_df) :
            raw_audio_df.iloc[i:i_next].to_pickle(('../Project/data/raw_audio_all_'+ str(k)+'.pickle'))
        else :
            raw_audio_df.iloc[i:len(raw_audio_df)].to_pickle(('../Project/data/raw_audio_all_'+ str(k)+'.pickle'))
            break

        i = i_next

def main_train_audio_features():
    '''
    - Function that allow the computation of all the features.
    - Process :
    1. Load the raw audio files.
    2. Features Extraction :
        2.1. Loading the Previously pickled raw audio file.
        2.2. Computing the MFCC of all the cutted version of the audio files.
        2.3. Saving them in a Pickled Pandas DataFrame
    '''
    audio_loaded_df = pd.read_pickle(('../Project/data/raw_audio_all_'+ str(1)+'.pickle'))

    for i in range(2,12):
        audio_loaded_df = audio_loaded_df.append(pd.read_pickle(('../Project/data/raw_audio_all_'+ str(i)+'.pickle')))

    # Optimal Parameters :
    N_MFCC = 10
    N_FFT =  int(2048/2)
    NUM_MFCCS_VEC = 20
    audio_loaded_df = audio_loaded_df.drop(2113).reset_index(drop=True)
    features_og = compute_mfcc_raw(audio_loaded_df,N_MFCC,N_FFT,NUM_MFCCS_VEC,cut=True)

    # Save features DataFrame as pickle
    features_og.drop(axis=1,columns=('audio')).to_pickle('./Features Data/cut_mfccs_all_raw_10_1028_20.pickle')
    features_og.head(2)


def load_audio_file(features_og):

    print("----- Start Importation -----")

    count_drop = 0
    audio_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('audio','raw',''),('audio','sr',''),('audio','cut','')]),index=features_og.index)

    for w in tqdm(range(len(features_og)),total=len(features_og),unit='waves'):

        audio, sampling_rate = librosa.load(features_og[('info','path')].iloc[w], sr=None, mono=True)

        clean_condition = (np.max(audio) != 0.0)

        if clean_condition:
            audio_df.loc[w,('audio','raw','')] = audio
            audio_df.loc[w,('audio','sr','')] = sampling_rate
            audio_df.loc[w,('audio','cut','')] = cut_signal(audio)
        else :
            count_drop += 1
            audio_df.drop(w)
            features_og.drop(w)


    audio_df = features_og.merge(audio_df,left_index=True,right_index=True)
    print("----- End Importation -----")
    print("Number of dropped signals :",count_drop)
    return audio_df


def compute_mfcc_raw(features_og,N_MFCC,N_FFT,NUM_MFCCS_VEC,cut=True):
    '''
    This function computes the raw MFCC parameters for and allow the choice of parameters
    '''

    stat_name= ['raw_mfcc']
    col_names = [('mfcc',stat_name[i],j) for i in range(len(stat_name))  for j in range(N_MFCC*NUM_MFCCS_VEC)]

    features_mfcc = pd.DataFrame(columns=pd.MultiIndex.from_tuples(col_names),index=features_og.index)
    # sorting the columns in order to improve index performances (see lexsort errors)
    features_mfcc.sort_index(axis=1,inplace=True,sort_remaining=True)

    # MFCC FEATURES :
    for w in tqdm(range(len(features_og)),total=len(features_og),unit='waves'):
        # Handling the cut version of the signal :
        if cut == True :
            audio = features_og.loc[w,('audio','cut','')]
        else :
            audio = features_og.loc[w,('audio','raw','')]

        sampling_rate = features_og.loc[w,('audio','sr','')]

        # Computing the MFCC for each signal :
        mfcc = librosa.feature.mfcc(y=audio,sr=sampling_rate, n_mfcc=N_MFCC, n_fft = N_FFT, hop_length = int(np.floor(len(audio)/NUM_MFCCS_VEC)))

        features_mfcc.loc[w, ('mfcc', 'raw_mfcc')] = mfcc[:,:-1].reshape(-1,)

    features_og = features_og.merge(features_mfcc,left_index=True,right_index=True)
    return features_og

def fit_and_test(clf, train_x, train_y, test_x, test_y):
    clf.fit(train_x, train_y)  
    predict_y = clf.predict(test_x)
    return predict_y

def adapt_labels(x_hat, class_names):
    # Real accuracy considering only the main words :
    class_names_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    mask_names_main = [True if name in class_names_list else False for name in class_names]
    index_names_main = [i for i in range(len(mask_names_main)) if mask_names_main[i] == True]
    inverted_index_names = dict(zip(index_names_main,range(len(index_names_main))))

    # Creating the label names :
    class_names_main = class_names[mask_names_main].tolist()
    class_names_main.extend(["unknown"])

    # Adapting the labels in the test and prediction sets :
    return np.array([inverted_index_names[int(x_hat[i])] if x_hat[i] in index_names_main else len(class_names_main)-1 for i in range(len(x_hat)) ]),class_names_main

def solve(Y_compr, M, L, alpha, beta):
    """Solves the above defined optimization problem t find an estimated label vector."""
    X = np.ones(Y_compr.shape)
    for i in range(Y_compr.shape[0]):
        Mask = np.diag(M[i,:])
        y_i_compr = Y_compr[i,:]
        X[i,:] = np.linalg.solve((Mask+alpha*L+beta),y_i_compr)
        
    return X


# pipeline function for semisupervised learning using graphs
def semisup_test_all_dataset(features_og, y, batch_size, NEIGHBORS, alpha, beta, iter_max, class_names):
    """Test semisupervised graph learning algorithm for entire dataset.
    - features_og : original copy of all MFCCs
    - batch_size : number of samples to be predict per iteration in main loop
    - NEIGHBORS : number of neirest neighbors in k-NN sparsification
    - alpha : hyper-parameter which controls the trade-off between the data fidelity term and the smoothness prio
    - beta : hyper-paramter which controls the importance of the l2 regularization for semi-supervised learning
    """
    accuracy_mat  = np.zeros((2,iter_max))
    
    for itr in tqdm(range(iter_max)):
        # Specify the number of datapoints that should be sampled in each class to build training and validation set
        train_size = 160
        valid_size = 1553

        train_x = np.array([])
        train_y = np.array([])

        valid_x = np.array([])
        valid_y = np.array([])

        for i in range(len(class_names)):
            class_index = np.where(y == (i+1))[0]
            random_index = np.random.choice(range(len(class_index)), size=train_size+valid_size, replace=False)

            train_x_class = class_index[random_index[:train_size]]
            train_y_class = y[train_x_class]
            train_x = np.append(train_x, train_x_class).astype(int)
            train_y = np.append(train_y, train_y_class).astype(int)

            valid_x_class = class_index[random_index[train_size:train_size+valid_size]]
            valid_y_class = y[valid_x_class]
            valid_x = np.append(valid_x, valid_x_class).astype(int)
            valid_y = np.append(valid_y, valid_y_class).astype(int)

        # Choose datapoints from validation set at random to form a batch
        potential_elements  = np.array(list(enumerate(np.array(valid_x))))
        indices = np.random.choice(potential_elements[:,0].reshape(-1,), batch_size, replace=False)
        # The batch index_variable contains the indices of the batch datapoints inside the complete dataset
        batch_index = potential_elements[:,1].reshape(-1,)[indices]
        
        # Build data matrix and normalize features
        X = pd.DataFrame(features_og['mfcc'], np.append(train_x, batch_index))
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        
        # Compute distances between all datapoints
        distances = spatial.distance.squareform(spatial.distance.pdist(X,'cosine'))
        n=distances.shape[0]

        # Build weight matrix
        kernel_width = distances.mean()
        W = np.exp(np.divide(-np.square(distances),kernel_width**2))

        # Make sure the diagonal is 0 for the weight matrix
        np.fill_diagonal(W,0)
        
        # compute laplacian
        degrees = np.sum(W,axis=0)
        laplacian = np.diag(degrees**-0.5) @ (np.diag(degrees) - W) @ np.diag(degrees**-0.5)
        laplacian = sparse.csr_matrix(laplacian)
        
        # Spectral Clustering --------------------------------------------------------------------------------
        eigenvalues, eigenvectors = sparse.linalg.eigsh(A=laplacian,k=25,which='SM')
        # Splitt Eigenvectors into train and validation parts
        train_features = eigenvectors[:len(train_x),:]
        valid_features = eigenvectors[len(train_x):,:]
        
        clf = QuadraticDiscriminantAnalysis()
        predict_y = fit_and_test(clf, train_features, train_y, valid_features, np.array(y[batch_index]))
        
        valid_y_adapted, class_names_main = adapt_labels(np.array(y[batch_index]),class_names)
        predict_y_adapted, class_names_main = adapt_labels(predict_y,class_names)
        accuracy_mat[0,itr] = np.sum(valid_y_adapted==predict_y_adapted)/len(valid_y_adapted)
        
        # Semi-Supervised Learning-----------------------------------------------------------------------------
        # Sparsify using k- nearest neighbours and make sure it stays symmetric
        # Make sure
        for i in range(W.shape[0]):
            idx = W[i,:].argsort()[:-NEIGHBORS]
            W[i,idx] = 0
            W[idx,i] = 0
        
        # Build normalized Laplacian Matrix
        D = np.sum(W,axis=0)
        L = np.diag(D**-0.5) @ (np.diag(D) - W) @ np.diag(D**-0.5)
        L = sparse.csr_matrix(L)
 
        # Build one-hot encoded class matrix
        Y_t = np.eye(len(class_names))[train_y - 1].T
        
        # Create Mask Matrix
        M = np.zeros((len(class_names), len(train_y) + batch_size))
        M[:len(train_y),:len(train_y)] = 1

        # Create extened label matrix and vector
        Y = np.concatenate((Y_t, np.zeros((len(class_names), batch_size))), axis=1)
        
        # Solve for the matrix X
        Y_hat = solve(Y, M, L,alpha = 1e-3, beta  = 1e-7)

        # Go from matrix X to estimated label vector x_hat
        y_predict = np.argmax(Y_hat,axis = 0)+np.ones(Y_hat[0,:].shape)
        
        # Adapt the labels, whee all words of the category "unknown" are unified
        y_predict_adapted, class_names_main = adapt_labels(y_predict,class_names)
        y_adapted, class_names_main = adapt_labels(np.array(y[batch_index]),class_names)

        # Compute accuracy in predicting unknown labels
        accuracy_mat[1,itr] = np.sum(y_predict_adapted[-batch_size:]==y_adapted)/batch_size

    return accuracy_mat
