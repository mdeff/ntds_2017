"""
This module contains all kinds of functions needed for execution of the notebook
"""

import scipy
import os
import json
import pandas as pd
import itertools
import numpy as np
import warnings
import requests
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from skimage.feature import local_binary_pattern
import sqlalchemy as sql
import random
from scipy import sparse, stats, spatial
from tqdm import tqdm_notebook


def featureToLaplacian(features, type_distance, laplacian_type, kernel, sparsification, thresh=None, neighbours=None):
    """ Pipeline that transforms features to laplacian

    Usage: [eigenvalues, eigenvectors, laplacian, weights] = featureToLaplacian(features, size, thresh)

    Input variables:
        features: dataframe containing the features and the nodes (countries)
        size: number of features
        type_distance: cosine or canberra (usually cosine distance)
        kernel: useful to choose when sparsification method is threshold-based. available kernel: cosine, triweight, gaussian
        sparsification: 'THRESHOLD' or 'NN'
        thresh: threshold to sparsify the network: to be defined if THRESHOLD method chosen
        neighbours: to be defined if NN method chosen.

    Output variables:
        eigenvalues: of the laplacian
        eigenvectors: of the laplacian
        laplacian: laplacian matrix corresponding to our topic
        weights: weight matrix
    """
   #FEATURE NORMALIZATION
    features -= np.nanmean(features,axis=0)
    features /= np.nanstd(features,axis=0)
 
    size=len(features.index)
    
    #computing the weights matrix (similarity network)
    distances=spatial.distance.squareform(spatial.distance.pdist(features,type_distance))
    
    if kernel == 'gaussian':
        #gaussian kernel
        kernel_width = distances.mean()
        weights=scipy.exp(-distances**2/kernel_width**2)
    elif kernel=='cosine':
        #cosine kernel
        weights=(np.pi/4)*np.cos((np.pi/2)*distances)
    elif kernel=='triweight':
        #triweight kernel
        weights=(35/32)*(1-(distances)**2)**3
    
    np.fill_diagonal(weights,0)
   
   
    #SPARSIFICATION:
    if sparsification == 'THRESHOLD':
        weights[weights < thresh*np.amax(weights)] = 0  # Sparse graph.
    elif sparsification=='NN':
        for i in range(len(weights)):
            sorted_w=np.argsort(weights[i])
            weights[i,(sorted_w[0:len(weights)-neighbours])]=0
            
    #MAKE MATRIX SYMMETRIC
    bigger= weights.transpose() > weights
    weights = weights - weights*bigger + weights.transpose()*bigger # Symmetric graph.
 
    D= np.zeros((size,size))
    degrees = np.sum(weights, axis=1) #dunno if degree must be integer number of connection, or sum of weights?\n",
    np.fill_diagonal(D,degrees)
    laplacian = D-weights #laplacian (not normalized)
    
    if laplacian_type=='normalized':
        # Compute the normalized laplacian
        D[np.diag_indices(size)] = 1/ (D.diagonal()**0.5) #we only apply it to the diagonal elements, otherwise division by zero --> error
        laplacian= np.dot(D, laplacian).dot(D)
        laplacian = sparse.csr_matrix(laplacian)

    
    #eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian, k=10, sigma=0)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian, k=3, which='SM')
    
    return eigenvalues, eigenvectors, laplacian, weights



def theme_selector(df,years,my_dict,text,comparison='and',to_print=0):
    # example : theme_selector(df = data, my_dict = code_to_indicator, years = years, text = ['hiv','tuberculosis'], comparison = 'and', to_print = 1)
    tuples_present = [x for x in df.columns]
    listOfKeys = []
    # CREATION OF LIST WITH THE INDICATORS SPECIFIED IN TEXT
    for what in text:
        keys = set([key for key, value in my_dict.items() if what.lower() in value.lower()])
        if keys: #not empty
            listOfKeys.append(keys)
        else:
            print('No indicator with word ''{}'' was found'.format(what))
    # IF THERE EXISTS SOME INDICATORS CONTINUE, OTHERWISE RETURN NULL
    if listOfKeys:
        if(comparison.lower()=='and'):
            finalKeys = set.intersection(*listOfKeys)
        elif(comparison.lower()=='or'):
            finalKeys = set.union(*listOfKeys)
        #Create all possible tuples
        # if only one year, convert it to list so it works for the combinations command
        if(isinstance(years, str)):
            years = [years]
        combinations = list(itertools.product(years,finalKeys))
        combinations = list(set.intersection(set(combinations),set(tuples_present)))
        # Buid final dataframe
        final = df[combinations]
        if(to_print):
            how_many_indicators = len(np.unique(final.columns.get_level_values(1)))
            how_many_years = len(np.unique(final.columns.get_level_values(0)))
            print('{} indicators correpond to your query (see below):'.format(how_many_indicators))
            if(how_many_years == 1):
                print('You selected the year {}'.format(years))
            else:
                print('You selected {} years'.format(how_many_years))
            print('---------------------------------------------')
            for indicator in np.unique(final.columns.get_level_values(1)):
                print('{} = {}'.format(indicator,my_dict[indicator]))
        return final
    else:
        return []

    
    
    
def numMissing(x):
    """function to count the number of non existing values in the dataset"""
    return sum(x.isnull())


def find_key(input_dict, value):
    """#function to retrieve the key from value:"""
    return next(k for k, v in input_dict.items() if v == value)
    
   
    
    
def cleanNaN (raw_df):
    """ clean a raw dataframe by removing all the NaN

    Usage: [clean_dataframe] = cleanNaN(raw_dataframe)

    Input variables:
        raw_df: raw_dataframe that contains the countries (rows) and the features (columns)

    Output variables:
        clean_df: see input variable despcription
    """
    #first step:cleaning the NaN values when the whole row or whole columns is empty:
    raw_df.dropna(axis=1, how='all' )
    raw_df.dropna(axis=0, how='all')
  

    while True:

        #removing indicators (columns)
        NaN_per_col = raw_df.apply(numMissing, axis=0).to_frame().T #axis=0 defines that function is to be applied on each column
        max_NaN_per_col = np.amax(NaN_per_col.iloc[0,:].values)
        min_NaN_per_col=np.amin(NaN_per_col.iloc[0,:].values)
        if max_NaN_per_col == min_NaN_per_col+2 or max_NaN_per_col == min_NaN_per_col+1 : 
            threshold1=len(raw_df.index)-max_NaN_per_col+1
        else:
            threshold1=len(raw_df.index)-max_NaN_per_col+3 #if rate of 2 here, check the condition at +1
        raw_df=raw_df.dropna(axis=1, thresh=threshold1)
        
            
        # check conditions to stop the while loop
        if raw_df.isnull().sum().sum() == 0:
            break
        
        
        #removing countries (rows)
        NaN_per_row = raw_df.apply(numMissing, axis=1).to_frame().T #axis=1 defines that function is to be applied on each row
        max_NaN_per_row = np.amax(NaN_per_row.iloc[0,:].values)
        threshold0=len(raw_df.columns)-max_NaN_per_row +1
        raw_df=raw_df.dropna(axis=0, thresh=threshold0)
        

        # check conditions to stop the while loop
        if raw_df.isnull().sum().sum() == 0:
            break
        
    return raw_df

def theme_selector_2(key_words, dictionary, code_indicators):
    
    theme_list=[]
    indicators=[]
                           
    #conversion from indicators code to string for the dataset given
    for ii in code_indicators:
        indicators.append(dictionary[ii])
        
    #looking for keywords
    for i in key_words:
        for j in indicators:
            if i in j and (not j in theme_list):
                theme_list.append(j)
            
    return theme_list

def group_data(data,interval,clean=None,what=None):
    """ Groups the data by time intervals

    Usage: final = group_data(data,interval,clean=None,what=None)

    Input variables:
        data: dataframe containing the data: countries x (years,indicators)
        interval: how many years to group together. If 5, will group 1960-1964, etc.
        clean: whether to clean the final data with cleanNaN, by default False
        what: which statistics to group the data, by default "mean", other possibilities are "median", "max" or "min"

    Output variables:
        final: dictionnary containing as keys the year ranges and as values the associated dataframe. For instance data['1960-1964'] contains the associated dataframe.
    """
    
    # Set default arguments
    if(what==None):
        what = "mean"
    if(clean==None):
        clean = False
    # Removes a weird date if found in the data
    if('Unnamed: 60' in np.unique(data.columns.get_level_values(0))):
        warnings.warn("\nThe following date was found: \'Unnamed: 60\', removing it from the data before proceeding...")
        data.drop(labels='Unnamed: 60',axis=1,level=0,inplace=True)
    
    # Defines the function we will use to aggregate values, by default mean
    def which_function(): 
        if(what.lower()=="mean"):
            return 0
        elif(what.lower()=="median"):
            return 1
        elif(what.lower()=="min"):
            return 2
        elif(what.lower()=="max"):
            return 3
        
    # Actual computation
    year_list=list(np.unique(data.columns.get_level_values(0))) #get years list
    how_many=int(np.floor(len(year_list)/interval)) #how many intervals we will have. if interval = 5. We will have data from 1960-1964, etc. which makes 56/5 ~= 11 year intervals
    final = dict() #final data storage. Better than creating dynamic variables. Use as final['1960-1964'] = dataframe for those years
    for i in tqdm_notebook(range(0,how_many)):
        key = year_list[i*interval]+'-'+year_list[(i+1)*interval-1]
        if(which_function()==0):
            value = data[year_list[i*interval:(i+1)*interval]].mean(axis=1,level=1).dropna(axis=1, how='all')
        elif(which_function()==1):
            value = data[year_list[i*interval:(i+1)*interval]].median(axis=1,level=1).dropna(axis=1, how='all')
        elif(which_function()==2):
            value = data[year_list[i*interval:(i+1)*interval]].min(axis=1,level=1).dropna(axis=1, how='all')
        elif(which_function()==3):
            value = data[year_list[i*interval:(i+1)*interval]].max(axis=1,level=1).dropna(axis=1, how='all')
        # Clean it with CleanNaN or not ? Takes a long time !
        if(clean):
            final[key] = cleanNaN(value)
        else:
            final[key] = value
        
    return final


def get_labels(countries, method=None, df=None):
    """ Get labels according to method

    Usage: labels, code_to_labels, colors = get_labels(method=None, df=None)

    Input variables:
        countries: list of countries. Typically data.index
        method: string of type 'incomeLevel', 'lendingType', 'region', 'migration'
            'incomeLevel' = (DEFAULT VALUE) 4 classes : low-income, lower-middle income, upper-middle income, higher-income
            'lendingType' = 3 classes : 'IDA', 'Blend', 'IBRD' ; IDA refers to The International Development Association (IDA).
            an international financial institution which offers concessional loans and grants to the world's poorest 
            developing countries; and IBRD refers to an international financial institution that offers loans to 
            middle-income developing countries
            'region' = 7 classes : East Asia and Pacific, Europe and Central Asia, Latin America & the Caribbean,
            Middle East and North Africa, North America, South Asia, Sub-Saharan Africa
            'migration' = 2 classes : positive migration, negative migration. 
        df: ONLY IF METHOD = 'migration'. Dataframe for net migration: typically data.xs('SM.POP.NETM',level=1,axis=1)
        

    Output variables:
        labels: dictionnary with countries as keys and labels as values: ex: labels['ABW'] = 'HIC' (for incomeLevel)
        code_to_labels: dictionnary with labels code explained, like code_to_labels['HIC'] = 'High income'
        colors: dictionnary with colors for networkx. Each label has a color
    """
    
    custom = False
    if(method==None):
        method='incomeLevel'
    elif(method.lower()=='migration' and df is None):
        raise ValueError('You did not input any dataframe ! see function help')
    elif(method.lower()=='migration' and df is not None):
        custom = True # custom labeling
        

    URL = 'http://api.worldbank.org/v2/countries?per_page=350&format=json'
    try:
        response=requests.get(URL,timeout=5).json()[1]
    except:
        raise ValueError('Could not make call to API!')
        
    if(not custom):
        if(method!='incomeLevel'):
            keys_labels = [d['id'] for d in response if d['id'] in countries] # only get labels for the countries we are interested in
            values_labels = [d[method]['id'] for d in response if d['id'] in countries]
            values_method = [d[method]['value'] for d in response if d['id'] in countries] 
            # Create labels dict: like labels['ABW'] = 'HIC'
            labels = dict(zip(keys_labels,values_labels))
            # Create labels dict explained, like 'HIC' : 'High income'
            code_to_labels = dict(zip(values_labels,values_method))  

            # Create colors dict,  to get the list of colors to color the nodes based on the label. for networkx      
            color = ['b','g','r','c','m','y','k','w']
            my_dict = dict(zip(np.unique(values_labels),color)) #my_dict['HIC'] = 'g'
            colors = [my_dict[k] for k in values_labels]
            colors = dict(zip(keys_labels,colors))
        else: # incomelevel
            labels = dict()
            colors = dict()
            labels_2 = pd.read_excel('./data/OGHIST.xls', sheetname=1, header=5,index_col=0)
            labels_2 = labels_2.iloc[5:]
            labels_2.drop(labels_2.columns[0], axis=1,inplace=True)
            labels_2.replace(to_replace=['H','UM','LM','L','..','LM*'], value=['HIC','UMC','LMC','LIC','NA','LMC'], inplace=True)
            labels_2 = labels_2.loc[countries]
            labels_2.fillna('NA',inplace=True)
            
            for i in list(labels_2.columns):
                labels_2_temp = labels_2[i]
                keys_labels = [d['id'] for d in response if d['id'] in countries]
                values_labels = [labels_2[i].loc[key] for key in keys_labels]   
                values_method = [d[method]['value'] for d in response if d['id'] in countries] 
                labels[str(i)] = dict(zip(keys_labels,values_labels))

# Create colors dict,  to get the list of colors to color the nodes based on the label. for networkx      
                color = ['b','g','r','c','m','y','k','w']
                my_dict = dict(zip(np.unique(values_labels),color)) #my_dict['HIC'] = 'g'
                _colors = [my_dict[k] for k in values_labels]
                colors[i] = dict(zip(keys_labels,_colors))
            code_to_labels = dict(zip(values_labels,values_method))
    else:
        labels = dict()
        colors = dict()
        warnings.filterwarnings("ignore",category =RuntimeWarning) #because issues a warning when doing nan >= 0
        for i in list(df.columns): #1992, 1997, etc.
            #df_temp = cleanNaN(df[i])
            df_temp = df[i]
        #df_temp=df
            keys_labels = [d['id'] for d in response if d['id'] in countries]
            values_labels_1 = (df_temp.values >= 0)    
            values_labels_2 = (df_temp.values < 0)    
            values_labels = [1 if i and not j else -1 if not i and j else 0 for i,j in zip(values_labels_1,values_labels_2)]

            values_method = ['positive' if i > 0 else 'none' if i==0 else 'negative' for i in values_labels]
            labels[i] = dict(zip(keys_labels,values_labels))
        #labels = dict(zip(keys_labels,values_labels))
            # Create colors dict,  to get the list of colors to color the nodes based on the label. for networkx      
            color = ['r','w','b','c','m','y','k','w']
            my_dict = dict(zip(np.unique(values_labels),color)) #my_dict['HIC'] = 'g'
            _colors = [my_dict[k] for k in values_labels]
            colors[i] = dict(zip(keys_labels,_colors))
            if(len(set(values_labels)) == 3):
                code_to_labels = dict(zip(values_labels,values_method))
        #colors = dict(zip(keys_labels,_colors))

    
    return labels, code_to_labels, colors


def first_clean(data):
    """ Clean the raw data, by removing weird dates and aggregated countries"""
    
    data.drop(labels='Unnamed: 60',axis=1,level=0,inplace=True)#drop year called 'Unnamed: 60'
    not_a_country=['ARB','CEB','EAP','EAR','EAS','ECA','ECS','EUU','EMU','FCS','HIC','HPC','LAC','LDC','LIC',\
                   'LMC','LCN','LMY','LTE','MEA','MIC','MNA','NAC','OED','PRE','PST','SAS','SSA','SSF','TEA','TEC','TLA','TSA',\
                   'TMN','TSS','UMC','WLD']
    data.drop(not_a_country,axis=0,inplace=True) #drop countries that are aggreggates
    data = data.loc[:,~data.columns.duplicated()] # remove duplicate columns
    
    return data



def make_json(weights,countries,labels,theme,year_range, name_classification, dico):
    """ make_json file for nice javascript visualisation

    Usage: make_json(weights,countries,theme,year_range, name_classification)

    Input variables:
        weights: the weight matrix, output of ftl
        countries: list of str indicating the countries for each index.
        labels: dict of the form: {'ABW': 'HIC', 'AFG': 'LIC', ...}. If the label has several years the year from the middle of the range will be chosen
        theme: str: Which theme between : demographics, education, health, social, economic, gas_production, technology
        year_range: complete str of the form : '1997-1999' 
        

    Output variables:
        NO OUTPUT: writes json files to be used for visualisation
    """
    
    # Make numpy weight matrix into dataframe with rows & cols as countries labels
    df = pd.DataFrame(weights,index=countries,columns=countries,dtype=np.float64)
    # If the labels are by year, select the middle year
    key = list(labels.keys())[-1]
    if(type(labels[key]) == dict): #check if labels dict contains a dict. If yes the labels is either income or migration
        year = year_range.split('-') # gives ['1997', '1999'] for instance
        interval = int(year[-1]) - int(year[0]) # for above exemple : interval = 2
        if(key == '2012'): # special case for migration
            present_years = np.linspace(int(year[0]), int(year[-1]), num=interval+1) # gets all the years in interval
            present_years = ["%g" % x for x in year] # converts them to str
            year =  list(set(present_years) & set(labels.keys()))
            if not year:
                print('Default value for migration since no label for this year range was found')
                year = '1992' # default value
                labels = labels[year]
            else:
                year = year[0]
                labels = labels[year]
        else:
            year = str(int(int(year[0]) + np.ceil(interval/2))) # for above example : year = '1998'
            if(year in labels.keys()):
                labels = labels[year]
            else:
                print('For theme: {} and year {}, no labels were present. Skipping...'.format(theme,year))
                labels = labels['1992'] # default value
    else: # if not, the label is world region
        pass
    
    # Get unique labels and convert them to int groups
    code_to_labels = dict()
    for i in range(0,len(set(labels.values()))):
        code_to_labels[list(set(labels.values()))[i]] = i
    
    # create 'links' dictionnary. Basically json indicating the values between countries, such as {"source": "ARE", "target": "HUN", "value": 0.7386523090926218}
    final_list = []
    for i in range(0,len(df.index)):
        for j in range(i,len(df.index)):
            second = dict()
            if(i!=j):
                second['source'] = df.index[i]
                second['target'] = df.index[j]
                second['value'] = df[df.index[i]].loc[df.index[j]]
                if(second['value'] > 0): #only add values that are present
                    final_list.append(second)

    first = dict()
    first['nodes'] = [{'group-name':value, 'group': code_to_labels[value], 'id':dico[key]} for (key,value) in labels.items()]
    first['links'] = final_list
    
    # create folder if it doesn't exist
    if not os.path.exists('./json'):
        os.makedirs('./json')
    # Write json
    filename = '{}{}{}'.format(theme,year_range, name_classification)
    with open('./json/{}.json'.format(filename), 'w') as outfile:
        json.dump(first, outfile, ensure_ascii=False)
        
        
        
        
def solve(y, alpha, mask, lapl, classes):
    """
    usage: x_pred, x_star = solve(y,alpha=1,M=M,lapl=laplacian)
    Solve the optimization problem.
    
    Parameters:
        y: the observations
        alpha: the balance between fidelity and smoothness prior.
    
    Returns:
        x_pred: the predicted class
        x_star: the solution of the optimization problem
    """
    
    I=np.identity(len(y)) #number of nodes
    first_term= mask + alpha*lapl
    x_star=np.linalg.solve( first_term , y)
    x_pred=smooth_to_discrete(x_star, len(classes))
    return x_pred, x_star


def smooth_to_discrete(signal, how_many):
    
    """ usage:  x_pred=smooth_to_discrete(x_star, len(data_classes))
    
    ALlows to transform the smooth solution into a disccrete signal
    inputs: signal: smooth solution
    how_many: number of classes (int) for example, if migration, nb of classes =2
    """
     
    x_pred=np.zeros(len(signal)) # x_pred is found by attributing the label depending on the sign of the smooth solution

    a=min(signal)
    b=max(signal)
    interval=(b-a)/how_many
    classes=np.linspace(0,how_many-1, how_many)

    boarder=[]
    for i in range(how_many):
        boarder.append(a+i*interval)
    boarder.append(b)
    for i in range (len(signal)):
        for j in range(how_many):
            if boarder[j] <= signal[i]<= boarder[j+1]:
                x_pred[i]=classes[j]
    return x_pred


def prepare_observations(p, graph, digit_labels):
    #Prepare observations, where p is the percentage of values to keep.
    rs = np.random.RandomState(42)
    M = np.diag(rs.uniform(size=graph.N) < p)
    return M.dot(digit_labels), M

def make_theme_list(which, key_words, dictionary, codes):
    
    """Allows to create the list of indicators referring to a theme
    usage: theme_list, theme_name= make_theme_list(education, educ_kw, code_to_indicator, indicators)"""
    
    liste=theme_selector_2(key_words, dictionary,codes)
    name=which
    return liste, name 
    

def converter(codes,dico):
    """Converts a list of codes into a list of names"""
    names=[]
    for i in codes:
        names.append(dico[i])
    return names


def converter_reverse(names, dico):
    """Converts a list of names into a list of codes"""
    codes=[]
    for i in names: #Here precise which theme we want to explore
        codes.append(find_key(dico, i))
    return codes
