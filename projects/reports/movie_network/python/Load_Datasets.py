#basic imports
import sys, os, copy
import numpy as np
import datetime as dt
import pandas as pd
#import libraries for file checking
from os.path import isfile, join, isdir

#importing json (data are mostly in JSON)
import json
def Load_Datasets(FileAddress_movies,FileAddress_credits):
    ##input: location of files
    ##output: pandas dataframe containing all information on movies and credits
    
    def InputFilesFound(FileAddress):
        ##input:  file location
        ##output: True if file found, False otherwise
        return isfile(FileAddress) and not isdir(FileAddress)
    
    def Transform_LoadJSON(dataframe):
        ##input:  dataframe
        ##output: datafreme which JSON columns transformed     
        #itterating through JSON columns and loading json 
        JSONcolumns = IdentifyJSONcolumns(dataframe)
        for column in JSONcolumns:
            dataframe[column] = dataframe[column].apply(json.loads)
        return dataframe,JSONcolumns
    
    def IdentifyJSONcolumns(dataframe):
        ##input: dataframe 
        ##output: list of columns containing JSON  
        #getting list of collumn names
        columns=list(dataframe)
        JSONcolumns=[]
        #itteration though columns to find those with JSON
        for column in columns:
            try:
                json.loads(dataframe[column][0]) 
            except: continue
            JSONcolumns.append(column)   
        #returning list of columns in which JSON format was found
        return JSONcolumns
    
    def JSONtoKeyList(JSONentry,key):
        ##input:json entry and string key to identify what should be read
        ##output:string list which can be converted to an array
      
            
        INNERentries = []
        for InnerEntry in JSONentry:
            #if key =='id':
                #print(InnerEntry)
            INNERentries.append(InnerEntry[key])
        
        if len(INNERentries)>0:
            if key =='gender' or key=='id':
                outcome=''
                for entry in INNERentries:
                    outcome = outcome+str(entry)+','
                outcome=outcome[:len(outcome)-1]      
                return  outcome
            else:
                return  ','.join(INNERentries)
        return ''
        
        
    
    def JSONtoNameList(JSONentry):
        ##input: entry (one line) from JSON one of JSON columns
        ##output: strings of entries separated by commas
        return JSONtoKeyList(JSONentry,'name')
    def JSONtoIDList(JSONentry):
        ##input: entry (one line) from JSON one of JSON columns
        ##output: strings of entries separated by commas
        return JSONtoKeyList(JSONentry,'id')
    
    def JSONtoGenderList(JSONentry):
        return JSONtoKeyList(JSONentry,'gender')
    
    def JSONtoJobsList(JSONentry):
        return JSONtoKeyList(JSONentry,'job')
    
    def JSONtoDepsList(JSONentry):
        return JSONtoKeyList(JSONentry,'department')
        
        
    def Transform_JSONcolumnsDecapsulation(dataframe):
        ##input: dataframe
        ##output: dataframe which JSON columns decapsulated
        
        def GetJSONkeys(JSONunit):
            ##input: JSON data unit
            ##output: List of keys in the JSON dictionary
            JSONkeys=[]
            for key in JSONunit:
                JSONkeys.append(key)
            return JSONkeys   
        #allowing for changes of passed datased
        dataframe.is_copy = False 
        #reading JSON format and columns
        dataframe,JSONcolumns = Transform_LoadJSON(dataframe)
        #transforming JSON columns to text columns
        movies = pd.DataFrame()
        for column in JSONcolumns:
            
            if not (column=='production_countries' or column=='spoken_languages'):
                dataframe[column+'_id'] =dataframe[column].apply(JSONtoIDList)
            dataframe[column] =dataframe[column].apply(JSONtoNameList)

        return dataframe

    #Loading movies from file to dataframe
    def Load_movies(FileAddress_movies):
        ##input:  movie dataset location
        ##output: pandas Frame containing information about movies
        #reading raw dataset
        df_movies = pd.read_csv(FileAddress_movies)
        #decaplsulating json, making columns from list keys
        df_movies = Transform_JSONcolumnsDecapsulation(df_movies)
        return df_movies
    
    def Load_credits(FileAddress_credits):
        ##input: credits file adress
        ##output: credit dataframe
        df_credits = pd.read_csv(FileAddress_credits)
        df_credits = Transform_LoadJSON(df_credits)
        credits = pd.DataFrame()
        credits['title']           = df_credits[0]['title']
        credits['actors']          = df_credits[0]['cast'].apply(JSONtoNameList)
        credits['actors_id']       = df_credits[0]['cast'].apply(JSONtoIDList)
        credits['actor_gender']    = df_credits[0]['cast'].apply(JSONtoGenderList)
        credits['crew_names']      = df_credits[0]['crew'].apply(JSONtoNameList)
        credits['crew_names_id']   = df_credits[0]['crew'].apply(JSONtoIDList)
        credits['crew_jobs']       = df_credits[0]['crew'].apply(JSONtoJobsList)
        credits['crew_departments']= df_credits[0]['crew'].apply(JSONtoDepsList)
        return credits
    
    N=0
    def Get_primes(entries):
        INNERentries = entries.split(",")[:N]
        outcome=''
        for entry in INNERentries:
            outcome = outcome+str(entry)+','
        outcome =outcome[:len(outcome)-1]     
        return outcome
    
    #assuring that both dataset exists
    assert InputFilesFound(FileAddress_movies),  "Movies  input file not found"
    assert InputFilesFound(FileAddress_credits), "Credits input file not found"
   
    Final_dataset=Load_movies(FileAddress_movies)
    Final_dataset.set_index('title', inplace=True)
   
   
    Credentials = Load_credits(FileAddress_credits)
    Credentials.set_index('title', inplace=True)
   


    Final_dataset=Final_dataset.join(Credentials)
    Final_dataset['primary_genre']                 = Final_dataset["genres"].apply(lambda x: x.split(",")[0])
    Final_dataset['primary_production_company']    = Final_dataset["production_companies"].apply(lambda x: x.split(",")[0])
    N=15
    Final_dataset['prime_actors']                  = Final_dataset["actors"].apply(Get_primes)
    N=6
    Final_dataset['prime_crew_names']              = Final_dataset["crew_names"].apply(Get_primes)
    N=20 
    Final_dataset['prime_keywords']                = Final_dataset["keywords"].apply(Get_primes)
    
    #returning final dataset
    return Final_dataset