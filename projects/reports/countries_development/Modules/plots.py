"""
This module contains all kinds of functions needed for plots of the exploratory part
"""

import numpy as np
import pandas as pd
import wbdata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from Modules.utils import *
from sklearn import preprocessing, decomposition
from scipy import sparse, stats, spatial
from pygsp import graphs, filters, plotting


def plotSparsityFull(data):
    # PLOT SPARSITY PATTERN OF THE WHOLE DATASET
    years = list(data.columns.levels[0][:-1])
    plt.figure(figsize=(10,5))
    plt.spy(data[years].notnull(),aspect='auto');
    plt.title('Sparsity pattern of the whole dataset',fontsize=11);
    plt.xlabel('Years x Indicators',fontsize=15);
    plt.ylabel('Countries',fontsize=15);
    plt.show()
    
    
def plotSparsityRange(data,how_many):
    years = list(data.columns.levels[0][:-1])
    plt.figure(figsize=(10,5))
    step = int(len(years)/how_many)
    a=0
    b=step-1
    for i in range(0,how_many):
        plt.subplot(1, how_many, i+1)
        plt.spy(data[years[a:b]].notnull(),aspect='auto');
        plt.title('{}-{}'.format(years[a],years[b]),fontsize=11);
        if(i==0):
            plt.xlabel('Years x Indicators',fontsize=15);
            plt.ylabel('Countries',fontsize=15);
        plt.xticks([])
        plt.yticks([])
        plt.show()
        a+=step
        b+=step
        
def plotNANperyear(data):
    years = list(data.columns.levels[0][:-1])
    indicators = list(data.columns.levels[1])
    # Distribution of NaN values across years
    nan_per_years = []
    for i in range(0,len(years)): 
        temp = data[years[i]].isnull().sum().sum()
        nan_per_years.append(temp)

    plt.figure(figsize=(10,5))
    plt.plot(range(0,len(years)),100*(nan_per_years/data['1960'].size));
    plt.title('Number of NaN values per year',fontsize=15);
    plt.xlabel('Years',fontsize=15);
    plt.ylabel('# NaN values (% of total)',fontsize=15);
    plt.xticks(np.arange(0,len(years),8), tuple(years[0::8]));
    plt.show()

def plotNANpercountry(data,labels):
    countries = list(data.index)
    # Checks which type of label it is, either income level, migration or region, and selects the last year for income level or migration
    key = list(labels.keys())[-1]
    if(type(labels[key]) == dict):
        if(key=='2016'):
            key=='2015'
        labels2 = labels[key]
    else:
        labels2 = labels

    # Checks how many nans per country
    how_many_nans = data.isnull().astype(int).values.sum(axis=1)
    # Sorts the countries from most nan to less nan
    ind = np.argsort(how_many_nans)[::-1]

    # Helper function to group labels, so that visualization is easier
    def group_labels(lab):
        code = dict()
        labels_list = list(set(lab.values()))
        if(('LIC' or 'HIC') in labels_list): # by income level
            code['H'] = 'High income'
            code['L'] = 'Low income'
            code['M'] = 'Middle income'
            code['OT'] = 'Other'
            for key, value in lab.items():
                if value == 'HIC':
                    lab[key] = 'H'
                elif value == 'LIC':
                    lab[key] = 'L'
                elif value == 'LMC':
                    lab[key] = 'M'
                elif value == 'UMC':
                    lab[key] = 'M'
                elif value == 'NA':
                    lab[key] = 'OT'
        elif(('ECS' or 'NAC' or 'LCN' or 'SAS') in labels_list): #by region
            code['AF'] = 'Africa'
            code['AS'] = 'Asia and Pacific'
            code['EU'] = 'Europe'
            code['AM'] = 'America'
            code['OT'] = 'Other'
            for key, value in lab.items():
                if value == 'EAS':
                    lab[key] = 'AS'
                elif value == 'ECS':
                    lab[key] = 'EU'
                elif value == 'LCN':
                    lab[key] = 'AM'
                elif value == 'MEA':
                    lab[key] = 'AF'
                elif value == 'NA':
                    lab[key] = 'OT'
                elif value == 'NAC':
                    lab[key] = 'AM'
                elif value == 'SAS':
                    lab[key] = 'AS'
                elif value == 'SSF':
                    lab[key] = 'AF'

        return lab,code

    labels3,code = group_labels(labels2)

    # Gets labels, sorts them, puts them in (int) form
    range_countries = [countries[k] for k in ind] # sorted country code (by most nans): ['MAF', 'SXM', 'IMN', 'MNP', 'VGB',...]
    range_labels = [labels3[key] for key in range_countries] # sorted labels of countries above : ['HIC', 'LIC', 'UMC', 'HIC', ...]
    labels_to_int = dict(zip(set(range_labels), range(0,8))) # dict with labels to int : {'HIC': 1, 'LIC': 0, ...}
    range_labels_int = [labels_to_int[k] for k in range_labels] # sorted labels (as int) of countries 

    # Helper function to get arrays of consecutive values, to plot colors on xaxis, used below
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    # Define cmap for coloring the labels
    cmap = plt.get_cmap('jet_r')
    color = cmap(np.linspace(0, 1.0, len(np.unique(range_labels_int))))

    # Line plot of number of nan values per country, with background colored according to country label
    plt.figure(figsize=(10,5))
    plt.plot(range(0,len(countries)),[100*(how_many_nans[k]/data.loc['ABW'].size) for k in ind])
    plt.title('Number of NaN values per country, sorted (background = country label)',fontsize=15);
    plt.xlabel('Countries',fontsize=15);
    plt.ylabel('# NaN values (% of total)',fontsize=15);
    plt.xticks([]);

    # Prepare patches (to color the background according to the country label), and the legend
    legends = []
    for i in range(0,len(np.unique(range_labels_int))): # Let's say we have 4 labels : i=0:3
        index_country_label = [k for k in range(0,len(range_labels_int)) if range_labels_int[k] == i] # we get the index of each country with label i
        index_country_label = consecutive(index_country_label) # we get the consecutive indexes. For instance consecutive([1,2,3,5,7,8,9]) = [[1,3],[5],[7,9]]
        patch = mpatches.Patch(color=color[i], label=code[find_key(labels_to_int, i)], alpha=0.3) # Colors the background of each country according to its label
        legends.append(patch)
        for j in range(0,len(index_country_label)):
            temp = len(index_country_label[j])
            plt.axvspan(index_country_label[j][0],index_country_label[j][temp-1], color=color[i], alpha=0.3, lw=2.0) 

    plt.legend(handles=legends,loc='upper center', bbox_to_anchor=(0.5, -0.06),
              fancybox=True, shadow=True, ncol=len(np.unique(range_labels_int)),fontsize='xx-small')

    plt.show()
    

def plotNANperindicator(data,background=False):
    # PLOT NUMBER OF NAN PER INDICATOR
    indicators = list(set(data.columns.levels[1]))
    # Checks how many nans per indicator
    how_many_nans = []
    for i in indicators:
        temp = data.xs(i,level=1,axis=1).isnull().astype(int).values.sum()
        how_many_nans.append(temp)
    # Sorts the indicators from most nan to less nan
    ind = np.argsort(how_many_nans)[::-1]
    
    if(background):
        #create labels2 dictt
        sources = wbdata.get_source(display=False)
        id_to_sourceName = dict(zip([k['id'] for k in sources],[k['name'] for k in sources]))

        all_indics = wbdata.get_indicator(display=False)

        indicator_to_id = dict(zip([k['id'] for k in all_indics],[k['source']['id'] for k in all_indics]))
        for i in (set(indicators) - set(indicator_to_id.keys())):
            indicator_to_id[i] = -1
        id_to_sourceName[-1] = 'NaN'
        # Gets labels, sorts them, puts them in (int) form
        range_indics = [indicators[k] for k in ind] # sorted indicator code (by most nans): 
        range_labels_int = [indicator_to_id[key] for key in range_indics] 
        range_labels = [id_to_sourceName[indicator_to_id[key]] for key in range_indics]
        # Define cmap for coloring the labels
        cmap = plt.get_cmap('jet_r')
        color = cmap(np.linspace(0, 1.0, len(set(range_labels_int))))

    # Helper function to get arrays of consecutive values, to plot colors on xaxis, used below
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    # Line plot of number of nan values per country, with background colored according to indicator label
    plt.figure(figsize=(10,5))
    plt.plot(range(0,len(indicators)),[100*(how_many_nans[k]/data.xs('SP.POP.TOTL',level=1,axis=1).size) for k in ind])
    if(background):
        plt.title('Number of NaN values per indicator, sorted (background = indicator label)',fontsize=15);
    else:
        plt.title('Number of NaN values per indicator, sorted',fontsize=15);
    plt.xlabel('Indicators',fontsize=15);
    plt.ylabel('# NaN values (% of total)',fontsize=15);
    plt.xticks([]);

    if(background):
        # Prepare patches (to color the background according to the indicator label), and the legend
        legends = []
        a=0
        for i in list(set(range_labels_int)): # Let's say we have 4 labels : i=0:3
            index_country_label = [k for k in range(0,len(range_labels_int)) if range_labels_int[k] == i] # we get the index of each indicator with label i
            index_country_label = consecutive(index_country_label) # we get the consecutive indexes. For instance consecutive([1,2,3,5,7,8,9]) = [[1,3],[5],[7,9]]
            patch = mpatches.Patch(color=color[a], alpha=0.3) # Colors the background of each indicator according to its label
            legends.append(patch)
            for j in range(0,len(index_country_label)):
                temp = len(index_country_label[j])
                plt.axvspan(index_country_label[j][0],index_country_label[j][temp-1], color=color[a], alpha=0.3, lw=2.0) 
                a+=1

                plt.legend(handles=legends,loc='upper center', bbox_to_anchor=(0.5, -0.06),
                           fancybox=True, shadow=True, ncol=int((1/5)*len(np.unique(range_labels_int))),fontsize='small')

    plt.show()
    
def pca_plot(agg_theme_data, period, theme, methode):      
        
    yr = period.split('-')[0] # gives ['1997', '1999'] for instance
    features=agg_theme_data[period][theme].copy(deep=True)
    features -= features.mean(axis=0)
    features /= features.std(axis=0)
    dict_labels, code_to_labels, colors = get_labels(features.index, method=methode, df=None)
    labels=list(dict_labels[yr].values())
    data_classes=list(set(labels))
    d = dict(zip(data_classes, range(0,len(data_classes))))
    converted_labels = list((d[i]) for i in labels)#Convert labels into number labels
    code_to_label2 = dict(zip(converted_labels,labels))
    plt.rcParams['figure.figsize'] = (17, 5);

    #ax = plt.axes(projection='3d', axisbg='white')
    features_pca = decomposition.PCA(n_components=2).fit_transform(features)
    incomeLevel = preprocessing.LabelEncoder().fit_transform(converted_labels)
    
    col = list(set(colors[int(yr)].values()))
    
    fig, ax = plt.subplots();
    for g in np.unique(converted_labels):
        ix = np.where(incomeLevel == g);
        ax.scatter(features_pca[ix,0], features_pca[ix,1], c=col[g],alpha=0.5,s=80,label=code_to_label2[g]);
    ax.legend();
    #plt.show()
    plt.title('Projection on the first two principal components for the period {} and the theme {}'.format(period,theme));


def plot_spectral(agg_theme_data, period, theme, methode):

        yr = period.split('-')[0] # gives ['1997', '1999'] for instance
    
        features=agg_theme_data[period][theme].copy(deep=True) #important to copy features because when re-running the cell
        thresh=0.4 #0.4 good theshold for cosine kernel
        neighbours=40 #40 good neighbors number
        eigenvalues, eigenvectors, laplacian, weights = featureToLaplacian (features,'cosine','normalized', kernel='cosine', sparsification='NN', thresh=thresh, neighbours=neighbours)

        #GROUND TRUTH
        dict_labels, code_to_labels, colors = get_labels(features.index, method=methode, df=None)
        labels=list(dict_labels[yr].values())
        data_classes=list(set(labels))
        d = dict(zip(data_classes, range(0,len(data_classes))))
        converted_labels = list((d[i]) for i in labels)#Convert labels into number labels

        #GRAPH EXPLORATION WITH PYGSP
        G = graphs.Graph(weights, gtype='countries')
        G.compute_laplacian(lap_type= 'normalized')
        laplacian=G.L.toarray()
        G.compute_fourier_basis(recompute=True)
        G.set_coordinates(kind= G.U[:,1:3]) #We project the data on the second and third eigenvectors

        # Play with the percentage of observed values.
        y, M = prepare_observations(p=0.25, graph=G, digit_labels= converted_labels)

        #SOLVING THE CLASSIFICATION PROBLEM
        x_pred, x_star = solve(y,alpha=0.5,mask=M,lapl=laplacian, classes=data_classes)

        # Error rate.
        err = np.count_nonzero(converted_labels - x_pred)
        print('For the period {} and the theme {}, we have {} errors ({:.2%})'.format(period,theme,err, err/G.N))

        G.plot_signal(np.array(converted_labels), vertex_size=100, show_edges=False)
        plt.title("Plot of the signal 'income level' over the nodes for the period {} and theme {}".format(period,theme))
        G.plot_signal(x_star, vertex_size=100, show_edges=False)
        plt.title("Visualization of the smooth solution")
        G.plot_signal(x_pred, vertex_size=100, show_edges=False)
        plt.title("Visualization of the predicted solution")
        plt.show()

def network_plot (agg_theme_data, period, theme, sparse, neigh,data):
    
    migration_list=[1992, 1997, 2002, 2007, 2012]
    yr = period.split('-')[0] # gives ['1997', '1999'] for instance
    yr_mig = min(migration_list, key=lambda x:abs(np.subtract(x,float(yr))))
    features=agg_theme_data[period][theme].copy(deep=True) #Important to duplicate without copying the object (reference)

    """parameters for weight matrix construction"""
    thresh=0.4 #0.4 good theshold for cosine kernel
    eigenvalues, eigenvectors, laplacian, weights = featureToLaplacian (features, 'cosine','normalized', kernel='cosine', sparsification='NN', thresh=thresh, neighbours=neigh)


    """VISUALIZATION WITH NETWORK X"""
    #methode='incomeLevel' #way of grouping the countries
    methodes=['incomeLevel', 'region', 'migration']
    G= nx.from_numpy_matrix(weights, create_using=None)
    labelsy=dict(zip(range(0,len(features.index)),list(features.index)))
    fixed_pos=nx.spring_layout(G)
    a=0
    plt.figure(figsize=(20,10))
    for methode in methodes:
        labels, code_to_labels, colors = get_labels(features.index, method=methode, df=data.xs('SM.POP.NETM',level=1,axis=1)) #if 'migration' used, use df=data.xs('SM.POP.NETM',level=1,axis=1)
        if methode=='incomeLevel':
            coloriage=colors[int(yr)]
            color_to_fullLabel = dict(zip(coloriage.values(),labels[yr].values()))
            legends_final = [color_to_fullLabel[k] for k in colors[int(yr)].values()]
            legends = [mpatches.Patch(color='{}'.format(i), label='{}'.format(code_to_labels[k])) for (i,k) in dict(zip(colors[int(yr)].values(),legends_final)).items()]
        elif methode=='migration':
            coloriage=colors['{}'.format(yr_mig)]
            color_to_fullLabel = dict(zip(colors['{}'.format(yr_mig)].values(),labels['{}'.format(yr_mig)].values()))
            legends_final = [color_to_fullLabel[k] for k in colors['{}'.format(yr_mig)].values()]
            legends = [mpatches.Patch(color='{}'.format(i), label='{}'.format(code_to_labels[k])) for (i,k) in dict(zip(colors['{}'.format(yr_mig)].values(),legends_final)).items()]
        else:
            coloriage=colors
            color_to_fullLabel = dict(zip(colors.values(),labels.values()))
            legends_final = [color_to_fullLabel[k] for k in colors.values()]
            legends = [mpatches.Patch(color='{}'.format(i), label='{}'.format(code_to_labels[k])) for (i,k) in dict(zip(colors.values(),legends_final)).items()]

        ax = plt.subplot('13{}'.format(a))
        nx.draw_networkx(G, pos=fixed_pos, labels=labelsy, node_color = [coloriage[k] for k in tuple(list(features.index))], \
                           width=0.2, with_labels=True, label = 'lol',ax=ax)

        plt.legend(handles=legends)
        
        a+=1
        plt.suptitle('Network representation of the countries for the period {} and the theme {} with the countries labeling'.format(period, theme))
        plt.title('{}'.format(methode))
    