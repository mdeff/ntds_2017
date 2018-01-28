import sys, os, copy
import numpy as np
import pandas as pd

#from multiprocessing import Pool
import multiprocessing
from multiprocessing import Pool, cpu_count
import functools

# !WARNING global viariables in use
Entries = 'blah'
myMovies=  pd.DataFrame()

def GetSingleEntryFrame(Comparator):
    def Transformer(Entrys):
        for Entry in Entrys.split(","):
            if Entry==Comparator:
                return 1
        return 0
    SingleEntryFrame = pd.DataFrame()
    SingleEntryFrame[Comparator]          = myMovies[Entries].apply(Transformer)
    Treshold = 2
    if Entries == 'keywords' or Entries == 'production_companies':
        Treshold == 2
    if SingleEntryFrame[Comparator].sum() <Treshold:
        return pd.DataFrame()
    return SingleEntryFrame

def join_dfs(ldf, rdf):
    return ldf.join(rdf, how='inner')

def applyParallel(df, func):
    multiprocessing
    with Pool(cpu_count()) as p:  
        ret_list = p.map(func, [Comparator for Comparator in GetEntryList(df)])
     
    print("Merging Started")
    Unified = pd.DataFrame()
    for df in ret_list:
        Unified[df.columns] = df[df.columns]
    return Unified

def GetEntryList(Movies):
    AllEntrys=""
    for Entrys in Movies[Entries]:
        AllEntrys+=Entrys
    EntryList =list(set(AllEntrys.split(",")))
    return EntryList

def TransformEntrys(Movies):
    
    EntrysFrame = applyParallel(Movies,GetSingleEntryFrame)
    return EntrysFrame 

def ProcessCategorical(Movies,feature):
    global Entries 
    Entries = feature
    global myMovies
    myMovies = Movies
    Frame=TransformEntrys(Movies)
    Frame.to_csv("../Datasets/"+str(feature)+".csv")
    print("Category processed") 
    return Frame