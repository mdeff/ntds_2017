from tqdm import tqdm
import json

import numpy as np
from numpy import pi
import pandas as pd
import scipy.sparse.linalg
from scipy import sparse, stats, spatial

import networkx as nx
from networkx.algorithms import community

import matplotlib.pyplot as plt
import seaborn as sns

import ephem
import reverse_geocoder as rg
from collections import Counter

from multiprocessing import Pool
from pygsp import graphs, filters, plotting

import random

def data_reduction_per_percentile(satcat_df,
                   TARGET_PERCENTILE_LAUNCH_SITES = 90,
                   TARGET_PERCENTILE_SOURCES = 90):
    """ Reduce the satcat_df using percentiles of the total data
    """
    launches_per_site = satcat_df.launch_site.value_counts()
    launches_per_source = satcat_df.source.value_counts()
    min_launch_nbr = np.percentile(launches_per_site, TARGET_PERCENTILE_LAUNCH_SITES)
    min_source_nbr = np.percentile(launches_per_source, TARGET_PERCENTILE_SOURCES)
    print("Min launch number: {}".format(min_launch_nbr))
    print("Min source number: {}".format(min_source_nbr))

    reduced_satcat_df = satcat_df
    for source, nbr in launches_per_source.iteritems():
        if nbr < min_launch_nbr:
            reduced_satcat_df = reduced_satcat_df.loc[~(reduced_satcat_df.source == source)]
    print("Unique sources: Initial: {}, Final: {}".format(len(satcat_df.source.unique()),len(reduced_satcat_df.source.unique())))

    for site, nbr in launches_per_site.iteritems():
        if nbr < min_launch_nbr:
            reduced_satcat_df = reduced_satcat_df.loc[~(reduced_satcat_df.launch_site == site)]
    print("Unique launch sites: Initial: {}, Final: {}".format(len(satcat_df.launch_site.unique()),len(reduced_satcat_df.launch_site.unique())))

    return reduced_satcat_df

def data_reduction_per_launch_site(satcat_df, LAUNCH_SITES = ["AFETR", "AFWTR"]):
    """ Reduce the satcat_df using the launch sites values """
    reduced_satcat_df = satcat_df
    to_keep = pd.DataFrame()
    for site in LAUNCH_SITES:
        to_keep = to_keep.append(reduced_satcat_df.loc[reduced_satcat_df.launch_site == site])
    reduced_satcat_df = reduced_satcat_df.loc[to_keep.index]
    return reduced_satcat_df

def normalize_features(features_df):
    """Normalize the features and drop any column containing null values"""
    features_df -= features_df.mean(axis=0)
    features_df /= features_df.std(axis=0)

    columns_to_drop = []
    for column in features_df.columns:
        if features_df[column].isnull().any():
            columns_to_drop.append(column)
    features_df = features_df.drop(columns_to_drop, axis=1)
    return features_df

def plot_weight_hist(weights, axes, name=None):
    """Plot an historigram from weight values"""
    axes[0].spy(weights)
    axes[1].hist(weights[weights > 0].reshape(-1), bins=50);
    if name:
        plt.savefig("fig/{}.png".format(name))

def check_symmetric(a, tol=1e-10):
    """Function to check if a is symmetric
        taken from: https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
    """
    return np.allclose(a, a.T, atol=tol)

def get_weights_from_distance(distances, kernel_width_percentile=0.5):
    # We use the percentile value to set the kernel width
    # It allows for more latitude than the mean and can easily be adapted
    kernel_width = np.percentile(distances, kernel_width_percentile)
    weights = np.exp((-np.square(distances))/kernel_width**2)

    # Set the diagonal weights to 0
    for index in range(len(weights)):
        weights[index][index] = 0
    return weights

def sparse_weights(weights: object, neighbors: object = 100, epsilon: object = 1e-8) -> object:
    """Function to sparsify the weight matrix
       It will set to zero the weights that are not in "neighbors"
       It will set to zero the weights that are smaller than "epsilon"
       It will ensure that the resulting matrix is symmetric
    """
    sorted_weights_indexes = np.argsort(weights, axis=1)

    # Set to zero the weights that are not in the NEIGHBORS
    for line in tqdm(range(len(sorted_weights_indexes))):
        for index in range(len(weights[0])-neighbors):
            weights[line][sorted_weights_indexes[line][index]] = 0.0

    # Filter weights that are too small
    # If we keep them ,we would have an highly connected graph
    # which is not what we want for the next sections
    for i in tqdm(range(len(weights))):
        for j in range(i):
            if weights[i][j] < epsilon:
                weights[i][j] = 0
            if weights[j][i] < epsilon:
                weights[j][i] = 0


    # Ensure symmetry
    for i in tqdm(range(len(weights))):
        # We need to check only the bottom triangle, because we do two checks
        # This reduces significantly the loop time
        for j in range(i):
            if weights[i][j] == 0 and weights[j][i] != 0:
                weights[i][j] = weights[j][i]

            if weights[i][j] != 0 and weights[j][i] == 0:
                weights[j][i] = weights[i][j]

    return weights

def get_weights_from_feature_dataframe(feature_df, kernel_width_percentile = 0.5, neighbors=100, epsilon=1e-8):
    """Functions to get sparcified weights directly from the feature dataframe"""
    distance_metric = "braycurtis"
    distances = spatial.distance.squareform(spatial.distance.pdist(feature_df, metric=distance_metric))
    weights = get_weights_from_distance(distances, kernel_width_percentile)
    weights = sparse_weights(weights, neighbors, epsilon)
    return weights

def create_graph_from_weights(weights, satcat_df):
    """Create a graph from the weights
       Each node will have the norad value as an argument
    """
    graph = nx.Graph(weights)
    norad_values = list(satcat_df.index)
    for node in graph.nodes():
        graph.node[node]['NORAD'] = norad_values[node]
    return graph

def get_nodes_per_site(graph, satcat_df):
    """Get the label of each node per launch site
       This assumes that there's at least two launch sites
    """
    nodes_per_site = {site:[] for site in satcat_df.num_launch_site.unique()}
    for node in graph.nodes():
        norad = graph.node[node]['NORAD']
        launch_site = satcat_df.loc[norad, "num_launch_site"]
        nodes_per_site[launch_site].append(node)
    return nodes_per_site

def draw_graph(graph, axes, satcat_df, name=None):
    nodes_per_site = get_nodes_per_site(graph, satcat_df)
    layout = nx.spring_layout(graph)

    for key in list(nodes_per_site.keys()):
        nx.draw_networkx_nodes(graph,
                               layout,
                               nodes_per_site[key],
                               node_color=plt.get_cmap('Set1')(key),
                               node_size=10,
                               ax=axes
                              )
    nx.draw_networkx_edges(graph,layout, width=0.1, ax=axes)
    if name:
        plt.savefig("fig/{}.png".format(name))

def remove_lonely_nodes(graph, minimum_degree = 0):
    nodes_to_drop = []
    for node in tqdm(graph.nodes()):
        if graph.degree(node) <= minimum_degree:
            nodes_to_drop.append(node)
    graph.remove_nodes_from(nodes_to_drop)
    return graph

def get_nodes_nbr(graphs):
    return [len(graph.nodes()) for graph in graphs]

def print_subgraphs_nodes_dist(subgraphs, axes, name=None):
    nodes_nbr = get_nodes_nbr(subgraphs)
    sns.distplot(nodes_nbr,kde=False, rug=True, ax=axes);
    if name:
        plt.savefig("fig/{}.png".format(name))

def print_subgraphs_network(subgraphs, satcat_df, name=None):
    """Use previously defined draw_graph function to draw
       and array of subgraphs
    """
    nbr_of_graphs = len(subgraphs)

    if nbr_of_graphs%2 == 1:
        nbr_of_subplots = nbr_of_graphs//2+1
    else:
        nbr_of_subplots = nbr_of_graphs//2

    fig, axes = plt.subplots(nbr_of_subplots,2, figsize=(20,nbr_of_graphs*1.5))

    for graph_index,graph in enumerate(subgraphs):
        draw_graph(graph, axes[graph_index//2][graph_index%2], satcat_df)

    if name:
        plt.savefig("fig/{}.png".format(name))

def get_big_subgraphs_index(subgraphs, max_nodes):
    """ Create an array with the index of the subgraphs
        that contain more than max_nodes nodes
    """
    big_subgraphs_index = []
    for index, graph in enumerate(subgraphs):
        if len(graph.nodes()) > max_nodes:
            big_subgraphs_index.append(index)
    return big_subgraphs_index

def get_graph_cliques(graph, smallest_clique = 10):
    """Determine the cliques of a graph and return
       an array of subgraphs that correspond to those cliques
    """
    c = list(community.k_clique_communities(graph,
                                            k=smallest_clique
                                           )
            )
    cliques = [clique for clique in c]
    subgraphs = [graph.subgraph(value) for value in cliques]
    return subgraphs

def create_unlabeled_label_df(satcat_df):
    """Create an unlabeled label dataframe from a satcat dataframe
    """
    number_of_sats = len(satcat_df)

    labels = np.full(number_of_sats, -1)
    is_labeled = np.full(number_of_sats, 0)

    real_labels = satcat_df.num_launch_site


    nodes_with_label = pd.DataFrame({"label":labels,
                                     "is_labeled":is_labeled,
                                     "NORAD":list(satcat_df.index),
                                     "real_label":real_labels.values})
    nodes_with_label = nodes_with_label.set_index("NORAD")
    return nodes_with_label

def label_nodes(label_df, percent_of_labeled):
    """This function will label of a fraction of the nodes"""
    number_of_sats = len(label_df)
    number_of_labeled = int(number_of_sats * (percent_of_labeled * 0.01))
    labeled_nodes  =  random.sample( list(label_df.index),
                                     k = number_of_labeled
                                   )
    for node in labeled_nodes:
        label_df.loc[node, "is_labeled"] = 1
        label_df.loc[node, "label"] = label_df.loc[node, "real_label"]
    return label_df

def create_labeled_df(satcat_df, percent_of_labeled):
    """Create a labeled dataframe"""
    label_df = create_unlabeled_label_df(satcat_df)
    label_df = label_nodes(label_df, percent_of_labeled)
    return label_df

def get_label_probs(label_df, subgraph):
    """Get the probability of each label for a specific subgraph"""
    subgraph_df = label_df.loc[[subgraph.node[node]['NORAD'] for node in subgraph.nodes()]]
    labeled_subgraph_df = subgraph_df[subgraph_df.is_labeled == 1]
    values_per_label = labeled_subgraph_df.real_label.value_counts()
    probability_dict = {}
    for label, count in values_per_label.iteritems():
        probability_dict[label] = count/len(labeled_subgraph_df)
    return probability_dict

def set_subgraph_label(label_df, subgraph, label):
    """Use to set all the labels of a subgraph to the same value"""
    for node in subgraph.nodes():
        norad = subgraph.node[node]["NORAD"]
        if label_df.loc[norad, "is_labeled"] == 0:
            label_df.loc[norad, "label"] = label
            label_df.loc[norad, "is_labeled"] = 1
    return label_df

def get_subgraphs_from_weights(weights, reduced_satcat_df, MAXIMUM_SUBGRAPH_NODES_PERCENT, SIZE_OF_SMALLEST_CLIQUE):
    """get the subgraphs using weights and reduced_satcat_df values
    """
    graph = create_graph_from_weights(weights, reduced_satcat_df)
    remove_lonely_nodes(graph)


    # Get subgraphs
    connected_subgraphs = []
    for subgraph in  nx.connected_component_subgraphs(graph):
        connected_subgraphs.append(nx.Graph(subgraph))


    maximum_subgraph_nodes = len(graph.nodes())*MAXIMUM_SUBGRAPH_NODES_PERCENT

    # Get the index of the big subgraphs
    big_subgraphs_index = get_big_subgraphs_index(connected_subgraphs,
                                                  maximum_subgraph_nodes
                                                 )
    # Segment the big subgraphs into cliques
    clique_subgraphs = []
    for subgraph_index in big_subgraphs_index:
        current_subgraph = connected_subgraphs[subgraph_index]
        current_subgraphs = get_graph_cliques(current_subgraph,
                                              smallest_clique=SIZE_OF_SMALLEST_CLIQUE
                                             )
        clique_subgraphs += current_subgraphs

    connected_subgraphs_no_big = [subgraph for index, subgraph in enumerate(connected_subgraphs)
                             if not index in big_subgraphs_index]
    subgraphs = clique_subgraphs + connected_subgraphs_no_big
    return subgraphs

def identify_from_prob(label_df, subgraph, probability_dict):
    """From the probability dictionnary, determine the label of the nodes"""
    if len(probability_dict) == 0: # Unknown
        label =  -1
    else:
        # NOTE: if there's equal probability, it will take one of them, which one isn't known
        label = max(probability_dict, key=probability_dict.get)

    label_df = set_subgraph_label(label_df, subgraph, label)
    return label_df

def identify_nodes_from_prob(label_df, subgraphs):
    """Identify every graph in subgraphs"""
    for subgraph in subgraphs:
        prob = get_label_probs(label_df, subgraph)
        label_df = identify_from_prob(label_df, subgraph, prob)
    return label_df

def get_labeled_df(satcat_df, subgraphs, percent_labeled):
    """Get a labeled dataframe"""
    labeled_df = create_labeled_df(satcat_df, percent_labeled)
    labeled_df = identify_nodes_from_prob(labeled_df, subgraphs)
    labeled_df = labeled_df[labeled_df.is_labeled == 1] # Keep only labeled nodes
    return labeled_df


def get_error_properties(label_df, sat_df, show=True):
    """Function to get various error properties from the label dataframe"""
    compared_label = label_df.label == label_df.real_label
    unknown_label = label_df.loc[label_df.label == -1]
    total_sat = len(sat_df)
    total_label = len(compared_label)
    total_good_label = compared_label.sum()
    total_bad_label = total_label - total_good_label
    total_unknown_label = len(unknown_label)
    good_classification_percent = total_good_label/total_label
    if show:
        print("Total Sats: {}, Labels:{}, Good:{}, Bad: {}, Unknown: {}, Percent: {}".format(total_sat,
                                                            total_label,
                                                            total_good_label,
                                                            total_bad_label,
                                                            total_unknown_label,
                                                            good_classification_percent
                                                           )
             )
    return {"total_sat" :total_sat,
            "total_label" : total_label,
            "total_good_label" : total_good_label,
            "total_bad_label" : total_bad_label,
            "total_unknown_label" : total_unknown_label,
            "good_classification_percent" : good_classification_percent
           }

def print_error_graph(error, file_name=None):
    """Function to print the errors in a graph"""
    x=list(error.keys())
    y1=[error[key]["good_classification_percent"]*100 for key in error.keys()]
    y6=[(1-error[key]["good_classification_percent"])*100 for key in error.keys()]
    y2=[error[key]["total_unknown_label"] for key in error.keys()]
    y3=[error[key]["total_good_label"] for key in error.keys()]
    y4=[error[key]["total_bad_label"] for key in error.keys()]
    y5=[error[key]["total_label"] for key in error.keys()]


    fig, ax1 = plt.subplots(figsize=(10,10))

    from matplotlib.ticker import AutoMinorLocator

    minorLocator = AutoMinorLocator()
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1.yaxis.set_minor_locator(minorLocator)
    ax1.grid(b=True, which='major', linestyle='-')
    ax1.grid(b=True, which='minor', linestyle='--')

    l1, = ax1.plot(x, y1, 'b', label="Good labels")
    l6, = ax1.plot(x, y6, 'm', label="Errors (bad label or unknown)")
    
    legend1 = plt.legend([l1, l6], ["Good labels", "Errors (bad label or unknown)"], loc=1)

    ax1.set_ylabel("Good labelling proportion (%)")
    ax1.set_xlabel("Labelized Nodes (%)")
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Nodes')
    ax2.set_yscale("log")

    l2, = ax2.plot(x, y2, 'c--', label="Total Unknown Label")
    l3, = ax2.plot(x, y3, 'g--', label="Total Good Label")
    l4, = ax2.plot(x, y4, 'r--', label="Total Bad Label")
    l5, = ax2.plot(x, y5, 'k--', label="Total Label")

    plt.title("Result analysis\nleft axis is for full lines, right for dotted lines")
    plt.legend([l2, l3, l4, l5], ["Total Unknown Label", "Total Good Label", "Total Bad Label", "Total Label"],  loc=3)
    if file_name:
        plt.savefig('fig/{}.png'.format(file_name))
    plt.show()

def calculate_all_values(satcat_df,
                         get_feature_dataframe,
                         REDUCE_PER_PERCENTILE= False,
                         REDUCE_PER_LAUNCH_SITE = True,
                         TARGET_PERCENTILE_LAUNCH_SITES = 90,
                         TARGET_PERCENTILE_SOURCES = 90,
                         LAUNCH_SITES = ["AFETR", "AFWTR"],
                         ONLY_PAYLOAD = True,
                         ONLY_OPERATIONAL = False,
                         SIZE_OF_SMALLEST_CLIQUE = 20,
                        ):
    """Function to get the reduced_satcat_df and the subgraphs from parameters"""
    # WEIGHTS PARAMETERS
    KERNEL_WIDTH_PERCENTILE = 0.5
    NEIGHBORS = 100
    EPSILON = 1e-8

    # GRAPH PARAMETERS
    MAXIMUM_SUBGRAPH_NODES_PERCENT = 0.20
    SIZE_OF_SMALLEST_CLIQUE = 20

    # Get reduced dataframe
    if REDUCE_PER_PERCENTILE:
        reduced_satcat_df = data_reduction_per_percentile(satcat_df, TARGET_PERCENTILE_LAUNCH_SITES, TARGET_PERCENTILE_SOURCES)
    elif REDUCE_PER_LAUNCH_SITE:
        reduced_satcat_df = data_reduction_per_launch_site(satcat_df, LAUNCH_SITES)

    if ONLY_PAYLOAD:
        reduced_satcat_df = reduced_satcat_df.loc[reduced_satcat_df.payload_flag == True]
    if ONLY_OPERATIONAL:
        reduced_satcat_df = reduced_satcat_df.loc[reduced_satcat_df.operational_status == "Operational"]

    print("getting feature vector")
    # Get feature vector
    feature_df = get_feature_dataframe(reduced_satcat_df, ONLY_PAYLOAD, ONLY_OPERATIONAL)

    print("getting weights")
    # Get weight matrix
    weights = get_weights_from_feature_dataframe(feature_df, KERNEL_WIDTH_PERCENTILE, NEIGHBORS, EPSILON)

    # Get subgraphs
    print("getting subgraphs")
    subgraphs = get_subgraphs_from_weights(weights, reduced_satcat_df, MAXIMUM_SUBGRAPH_NODES_PERCENT, SIZE_OF_SMALLEST_CLIQUE)

    del weights
    del feature_df

    result_dict = {"reduced_satcat_df":reduced_satcat_df, "subgraphs":subgraphs}
    return result_dict

def calculate_error(reduced_satcat_df, subgraphs, values = [0, 1, 2, 5, 10, 20, 40, 60, 80, 90, 100]):
    """Calculate the resulting error for a given result"""
    # Get labels
    error = {}
    for percent_labeled in values:
        labeled_df = get_labeled_df(reduced_satcat_df, subgraphs, percent_labeled)
        error[percent_labeled] = get_error_properties(labeled_df, reduced_satcat_df, show=False)
    for key in error.keys():
        print(key, ":", error[key]["good_classification_percent"])
    return error
