import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def cost(x):
    costs = np.zeros(Movies.shape[0])
    current_film = Movies.iloc[x]
    genres_current = get_genres(current_film)
    kw_current = get_keywords(current_film)
    for j in range(x,Movies.shape[0]):
        cost = 0
            
        b_film = Movies.iloc[j]
        genres_b = get_genres(b_film)
        #First we only select the first genre to determine the similarity because it's more important that the other genre.
        if len(genres_current) > 0  and len(genres_b) > 0:
            
            if (genres_current[0] == genres_b[0]):
                cost += first_genre
            #This give us the number of similar genres. We pop the first one because we already compare them.
            cost += np.sum(np.in1d(genres_current,genres_b.pop(0),assume_unique='True')) * second_genre


        kw_b = get_keywords(b_film)
        #This give us the number of similar keywords.
        cost += np.sum(np.in1d(kw_current,kw_b,assume_unique='True')) * keyword_cost
        costs[j] = cost
    return costs

def get_genres(film):
    genres = str(film['genres'])
    if genres == 'nan':
        return[]
    else:
        genres = genres.split(",")
    return genres

def get_keywords(film):
    kw = str(film['keywords'])
    if kw == 'nan':
        return[]
    else:
        kw = kw.split(",")
    return kw

def vote_ratio(x,costs):
    vote_x = Movies.iloc[x]['vote_average']
    for j in range(0,Movies.shape[0]):
        vote_j = Movies.iloc[j]['vote_average']
        costs[i,j] = costs[i,j] * vote_j / vote_x


if __name__ == '__main__':
    #Constant definition
    
    #Cost added if the first genre is similar between two films
    first_genre = 5
    #Cost added if the secondary genre is similar between two films
    second_genre = 1
    #Cost added by similar keyword identical between two films
    keyword_cost = 1
    
    usefull_columns = ['genres','keywords','vote_average']
    Movies = pd.read_csv("../Datasets/Transformed.csv",usecols=usefull_columns)
    Movies = Movies.loc[Movies['vote_average'] > 0]
    
    with Pool(cpu_count()) as p:

        r = list(tqdm(p.imap(cost, range(0,Movies.shape[0])), total=Movies.shape[0]))
        costs = np.array(r)
        
        costs = costs + costs.T
        
        r = list(tqdm(p.imap(vote_ratio, [range(Movies.shape[0],costs)]), total=Movies.shape[0]))
        costs = np.array(r)
        
        np.savez_compressed("../Datasets/costs_parallelized.npz", costs, costs = costs)
