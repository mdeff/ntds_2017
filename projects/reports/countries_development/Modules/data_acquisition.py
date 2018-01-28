import os
import numpy as np
import pandas as pd
import wbdata
import requests
import time  
import pickle
from tqdm import tqdm, tqdm_notebook


def data_format(df,dict_name_to_iso):
    """ Format data acquired with get_data, into a nice pandas dataframe

    Usage: df = data_format(df, dict_name_to_iso)

    Input variables:
        df: dataframe from get_data
        dict_name_to_iso: dictionnary of country name to ISO3 code

    Output variables:
        df: nicely formatted dataframe. With rows as ISO3 codes and columns as multiindex of year,indicator
    """
    
    # CLEAN DATAFRAME (remove aggregate year values) & ARRANGE
    # Removes aggregate year values
    df = df[df['Date'].map(len) < 5]
    # Removes a weird 'performance' indicator
    df = df[df['Indicator Code'] != 'Performance.']
    # Reshapes the table + changes index from country name to country iso3code
    df = df.pivot_table(df,index=df.index, columns=['Date','Indicator Code'], dropna=False)
    temp = df.index.to_series().map(dict_name_to_iso)
    df.set_index(temp,inplace=True)
    # Removes the rows (countries, i.e. nodes) that are not of interest, i.e. those that are not present in the initial data from kaggle
    df = df[df.index.notnull()]
    df = df.rename(columns={'Values':''})
    # Remove the first level that is useless
    df.columns = df.columns.droplevel(0)
    return df


def get_data(code_to_indicator, how_many_chunks, all_indicators):
    """ Get data from world bank.

    Usage: get_data(code_to_indicator, how_many_chunks, all_indicators)

    Input variables:
        code_to_indicator: dict of indicator code to indicator name
        how_many_chunks: separate indicators into 'how_many_chunks' so that if someting fails during the call, some of the data is saved
        all_indicators: list of all indicators we want to retrieve, typically 'wbdata.get_indicator(display=False)'

    Output variables:
        nothing: writes into a ./data folder where information is stored. The .pkl need later to be merged
    """
    
    # CREATE FOLDER TO STORE DATA IF NOT PRESENT ALREADY
    if not os.path.exists('./data'):
        os.makedirs('./data')

    URL = 'http://api.worldbank.org/v2/countries/all/indicators/{}?per_page={}&date=1960:2015&format=json'
    print('Starting...')
    for chunks in range(0,how_many_chunks):
        finalDF = pd.DataFrame()
        counter = -1 #counter of dropped indicators. -1 because of weird performance indicator named "Performance."
        for indicator in tqdm_notebook(all_indicators[chunks::how_many_chunks],desc='Indicators'):
            RETRIES = 0
            while(RETRIES < 5):
                try:
                    # To know if the indicator is empty or has some data. If it is empty -> next.
                    pages=requests.get(URL.format(indicator,1),timeout=5).json()
                    pages=pages[0]['total']
                    if(pages != 0):
                        dat=requests.get(URL.format(indicator,pages),timeout=60).json()
                        INDEX = [d['country']['value'] for d in dat[1]]
                        FIRST_LEVEL = [d['date'] for d in dat[1]]
                        SECOND_LEVEL = [d['indicator']['id'] for d in dat[1]]
                        VALUE = [d['value'] for d in dat[1]]
                        # UPDATE INDICATORS DICT
                        INDIC = dat[1][0]['indicator']['id']
                        INDIC_NAME = dat[1][0]['indicator']['value']
                        correspondance_indicator[INDIC]=INDIC_NAME
                        # CREATE DATAFRAME
                        temp = pd.DataFrame(list(zip(FIRST_LEVEL,SECOND_LEVEL,VALUE)),
                                            columns = ['Date','Indicator Code','Values'],index=INDEX)
                        temp.index.name = 'Country Code'
                        finalDF = finalDF.append(temp)
                    else:
                        counter+=1
                    break
                except requests.exceptions.Timeout:
                    RETRIES += 1
                    print('Timeout exception, retrying...')
                    time.sleep(60) 
                except:
                    RETRIES += 1
                    print('{}: Retrying for {}'.format(RETRIES,indicator))
                    time.sleep(60) 
        print('There are {0}/{1} ({2:.2f}%) potential new indicators'.
              format(len(all_indicators[chunks::how_many_chunks])-counter,len(all_indicators[chunks::how_many_chunks]),
                     100*(len(all_indicators[chunks::how_many_chunks])-counter)/len(all_indicators[chunks::how_many_chunks])))
        finalDF.to_pickle('./data/data{}.pkl'.format(chunks),compression='gzip')
        with open('./data/dict{}.pickle'.format(chunks), 'wb') as handle:
            pickle.dump(correspondance_indicator, handle, protocol=pickle.HIGHEST_PROTOCOL)

