#%matplotlib inline
#importing all relevant packages
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pygsp import graphs, filters, plotting
import pickle
import community
from scipy import sparse, stats, spatial
from SongJamAnalysis import *


plt.rcParams['figure.figsize'] = (10, 10)
plotting.BACKEND = 'matplotlib'

#Setting style for plotting
sns.set_style('whitegrid')
sns.set_palette('cubehelix',3)

'''
This function draws graphs which show the spectral analysis of the signal(spread over tim of jam) on the network.
This function is not modular and should only be used in the context it has been used in the notebook.
'''
def getInsightSongJam(artist, song, fig, axes, graph, data):
    if not(nx.is_connected(graph)):
        #keeping the largest connected component
        graph=sorted(nx.connected_component_subgraphs(graph), key = len, reverse=True)
        graph = graph[0]
        print('\nThe weigted network Gcc has {} nodes.'.format(len(graph.nodes())))
        print('The weigted network Gcc has {} edges.'.format(graph.size()))
        print('The nodes in Gcc has an average degree of {0:.2f}.'.format(sum(list(dict(nx.degree(graph,weight='weight')).values()))/len(graph.nodes())))
    #Get relevant data
    data_grimes = data.loc[(data.artist==artist) & (data.title==song)]
    #keep only two columns
    data_grimes = data_grimes[['user_id','creation_date']]
    #keep only ones in the node list
    data_grimes = data_grimes[(data_grimes.user_id.isin(graph.nodes()))]
    #change format to datetime
    data_grimes['creation_date'] = data_grimes['creation_date'].apply(pd.to_datetime)
    data_grimes.drop_duplicates(subset='user_id', inplace=True)
    data_grimes.reset_index(inplace=True)
    #initialize signal
    data_grimes['signal'] = 0
    data_grimes.loc[0, 'signal'] = 1

    #add signal has amount of days from the day of the first jam
    for i in range(1, data_grimes.shape[0]):
        a = data_grimes['creation_date'].iloc[i] - data_grimes['creation_date'].iloc[0]
        b = data_grimes['creation_date'].iloc[0] - data_grimes['creation_date'].iloc[i]
        data_grimes.loc[i, 'signal'] = a.days*24 + a.seconds/3600

    #create graph
    graph_f = graphs.Graph(nx.adjacency_matrix(graph))
    #sort the dataframe to be in the same order as the nodes
    data_grimes['user_cat'] = pd.Categorical(data_grimes['user_id'],categories=list(graph.nodes()),ordered=True)
    data_grimes.sort_values('user_cat', inplace=True)
    #compute fourier
    graph_f.compute_laplacian('normalized')
    graph_f.compute_fourier_basis()


    #plot the signal
    signal = data_grimes['signal'].values
    graph_f.set_coordinates(graph_f.U[:, [1, 2]])
    graph_f.plot_signal(signal,vertex_size=10, ax=axes[1])
    scale = 1
    plt.axis([-scale,scale,-scale,scale])

    axes[0].plot(np.abs(graph_f.gft(signal)))
    axes[0].set_xlabel('Laplacian Eigenvalues')
    axes[0].set_ylabel('Fourier domain response')
    axes[0].set_title('Implusion signal')

    # Compute the signal smoothness with gradient
    smoothness = signal.T @ graph_f.L @ signal / np.linalg.norm(signal)**2
    print("Signal gradient: %f" % smoothness)

    # Compute the suffles signal smoothness with gradient
    smoothness = 0
    for i in range(5):
        shuffled_signal = data_grimes.sample(frac=1)['signal'].values
        smoothness += shuffled_signal.T @ graph_f.L @ shuffled_signal / (5 * np.linalg.norm(shuffled_signal)**2)
    print("Shuffled signal gradient: %f" % smoothness)


'''
This function builds the network which contains only the people having jammed the targeted song.
This function is not modular and should only be used in the context it has been used in the notebook.
'''
def Create_network_jam(artist, title, net, groups, data, merge):
    data_network = data.loc[(data.artist==artist) & (data.title==title)]
    nodes_network = data_network['user_id']
    G = nx.Graph()
    G.add_nodes_from(nodes_network.values)

    #adding nodes and edges of followers
    net_jammers = net[(net['follower_user_id'].isin(nodes_network.values)) & (net['followed_user_id'].isin(nodes_network.values))]
    for i in range(len(net_jammers)):
        if net_jammers['followed_user_id'].iloc[i] != net['follower_user_id'].iloc[i]:
            G.add_edge(net_jammers['followed_user_id'].iloc[i],net_jammers['follower_user_id'].iloc[i],weight=1)

    #adding weights
    data_network = data.loc[(data.artist==artist) & (data.title==title)]
    nodes_network = data_network['user_id']
    jammers = merge[(merge['user_id_x'].isin(nodes_network.values)) & (merge['user_id_y'].isin(nodes_network.values))]
    grouped_jammers = jammers.groupby(['user_id_x','user_id_y']).count()
    for i in range(len(grouped_jammers)):
        if grouped_jammers.iloc[i].name[0] in G.nodes and grouped_jammers.iloc[i].name[1] in G.neighbors(grouped_jammers.iloc[i].name[0]):
            G[grouped_jammers.iloc[i].name[0]][grouped_jammers.iloc[i].name[1]]['weight'] += grouped_jammers.iloc[i]['jam_id']

    return G


'''
This function builds the network which contains only the people having jammed the targeted song.
The difference with the function above is that it also takes the neighbors of these people in the graph.
This function is not modular and should only be used in the context it has been used in the notebook.
'''
def Create_network(artist,title, data, net, merge):
    data_network = data.loc[(data.artist==artist) & (data.title==title)]
    nodes_network = data_network['user_id']
    G = nx.Graph()
    G.add_nodes_from(nodes_network.values)

    #adding nodes and edges of followers
    net_jammed = net[net['followed_user_id'].isin(nodes_network.values) & \
                     (net['follower_user_id'] != net['followed_user_id'])]
    for i in tqdm(range(len(net_jammed))):
        G.add_node(net['follower_user_id'].iloc[i])
        G.add_edge(net['followed_user_id'].iloc[i],net['follower_user_id'].iloc[i],weight=1)

    #adding weights
    jammers = merge[merge['user_id_x'].isin(list(G.nodes())) & (merge['user_id_y'].isin(list(G.nodes())))]
    grouped_jammers = jammers.groupby(['user_id_x','user_id_y']).count()
    for i in tqdm(range(len(grouped_jammers))):
        if grouped_jammers.iloc[i].name[1] in G.neighbors(grouped_jammers.iloc[i].name[0]):
            G[grouped_jammers.iloc[i].name[0]][grouped_jammers.iloc[i].name[1]]['weight'] += \
            np.log(grouped_jammers.iloc[i]['jam_id'])

    return G
'''
This function draws graphs which show the spectral analysis of the signal(spread over tim of jam) on the network.
This function is not modular and should only be used in the context it has been used in the notebook.
'''
def create_signal(network, artist, title, data):
    network_nodes = network.nodes()
    #Get relevant data
    data_n = data[(data.artist==artist) & (data.title==title)]
    #keep only two columns
    data_n = data_n[['user_id','creation_date']]
    #keep only ones in the node list
    data_n = data_n[(data_n.user_id.isin(network_nodes))]
    #initialize signal
    data_n['signal'] = 0
    data_n['signal'].iloc[0] = 1
    #change format to datetime
    data_n['creation_date'] = data_n['creation_date'].apply(pd.to_datetime)

    #add signal as amount of days from the day of the first jam
    for i in tqdm(range(1,len(data_n['user_id']))):
        a = data_n['creation_date'].iloc[i]-data_n['creation_date'].iloc[0]
        data_n['signal'].iloc[i] = a.days

    #adding extra data, the nodes that are user that didnt share the song
    data_n_extra = pd.DataFrame(columns=data_n.columns)
    data_n_extra['user_id'] = network_nodes
    #set their signal
    data_n_extra['signal'] = 2000
    #append all data
    data_n = data_n.append(data_n_extra)

    #drop duplicates, but since the ones in the signal are at the top they will be kept
    data_n = data_n.drop_duplicates('user_id',keep='first')

    #sort the dataframe to be in the same order as the nodes
    data_n['user_cat'] = pd.Categorical(data_n['user_id'],categories=list(network.nodes()),ordered=True)
    data_n = data_n.sort_values('user_cat')

    return data_n

'''
This function plots comparison between the music genre jammed by the different communities of a graph.
This function is not modular and should only be used in the context it has been used in the notebook.
'''
def compareCommunitiesTaste(communities, genres):
    comms_df = pd.DataFrame.from_dict(communities, 'index')
    comms_df.columns = ['community']
    comms_df.community = comms_df.community.astype(int)
    comms_df['col'] = 1

    genres = genres.merge(comms_df, left_index=True, right_index=True)

    biggest_comms = genres.groupby('community').sum().sort_values(by='col', ascending=False).iloc[:15]
    biggest_comms = biggest_comms.divide(biggest_comms.col.values, axis=0)

    # Taste comparison
    fig, ax = plt.subplots(figsize=(30,30))
    biggest_comms[biggest_comms.columns[:-1]].plot.barh(stacked=True, ax=ax, colormap='Paired')
