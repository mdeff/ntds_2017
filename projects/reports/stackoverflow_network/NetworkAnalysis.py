import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import linregress

def analyze_file(path):
    network_path = os.path.join("Tags", path)
    network = nx.read_edgelist(network_path,nodetype=int, data=(('time',int),('votes_q', int),('votes_a', int),('accepted', bool)))
    directed_network = nx.read_edgelist(network_path,create_using=nx.DiGraph(),nodetype=int, data=(('time',int),('votes_q', int),('votes_a', int),('accepted', bool)))
    analyze_network(network, directed_network)

def analyze_basic_file(path):
    network_path = os.path.join("Tags", path)
    network = nx.read_edgelist(network_path,nodetype=int, data=(('time',int),('votes_q', int),('votes_a', int),('accepted', bool)))
    analyze_basic(network)


def analyze_network(network, directed_network):
    """Executes all analysis functions for the given network"""
    analyze_basic(network)
    analyze_degrees(network, directed_network)
    #analyze_ERBA(network)
    #analyze_clustering(network)
    analyze_attributes(network)
    analyze_scale_free(network, directed_network)
    analyze_hubs(network, directed_network)


def get_number_nodes(network):
    return len(network.nodes())

def get_number_edges(network):
    return len(network.edges())

def get_number_connected_components(network):
    return nx.number_connected_components(network)

def get_number_self_loops(network):
    return network.number_of_selfloops()

def get_size_giant_component(network):
    comps = list(nx.connected_component_subgraphs(network))
    max_comp = max(comps, key=len)
    return len(max_comp.nodes())

def get_cluster_coefficient(network):
    try:
        return nx.average_clustering(network)
    except ZeroDivisionError:
        return 0

def get_avg_degree(network):
    return np.asarray(list(network.degree(network.nodes()).values())).mean()
    
def get_avg_in_degree(network):
    return np.asarray(list(network.in_degree(network.nodes()).values())).mean()
    
def get_avg_out_degree(network):
    return np.asarray(list(network.out_degree(network.nodes()).values())).mean()
    
def get_max_degree(network):
    try: 
        return max(list(network.degree(network.nodes()).values()))
    except ValueError:
        # empty network
        return 0

def plot_degree_hist(network):
    """Plots the degree distribution as histogram"""
    degrees_dict = network.degree()
    degrees = np.asarray([ [key,degrees_dict[key]] for key in degrees_dict])
    plt.yscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel("Degree")
    plt.hist(degrees[:,1])
    plt.show()
    
def plot_degree_scatter(network):
    """Plots the degree distribution as scatter plot"""
    degrees_dict = network.degree()
    degrees = np.asarray([ [key,degrees_dict[key]] for key in degrees_dict])
    degrees_cleaned = [v for v in degrees if v[1] != 0]
    unique, counts = np.unique(degrees_cleaned, return_counts=True)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel("Degree")
    plt.scatter(unique, counts)    
    plt.show()

def plot_in_degree_hist(network):
    """Plots the incoming degree distribution as histogram"""
    degrees_in_dict = network.in_degree()
    degrees_in = np.asarray([ [key,degrees_in_dict[key]] for key in degrees_in_dict])
    plt.yscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel("In coming Degree")
    plt.hist(degrees_in[:,1])
    plt.show()
    
def plot_in_degree_scatter(network):
    """Plots the incoming degree distribution as scatter plot"""
    degrees_in_dict = network.in_degree()
    degrees_in = np.asarray([ [key,degrees_in_dict[key]] for key in degrees_in_dict])
    degrees_in_unique = np.unique(degrees_in[:,1])
    degrees_in_unique = np.array([v for v in degrees_in_unique if v != 0])
    degrees_in_cleaned = [v for v in degrees_in if v[1] != 0]
    unique, counts = np.unique(degrees_in_cleaned, return_counts=True)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel("In coming Degree")
    plt.scatter(unique, counts)  
    plt.show()

def plot_out_degree_hist(network):
    """Plots the outgoing degree distribution as histogram"""
    degrees_out_dict = network.out_degree()
    degrees_out = np.asarray([ [key,degrees_out_dict[key]] for key in degrees_out_dict])
    plt.yscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel("Out going Degree")
    plt.hist(degrees_out[:,1])
    plt.show()
    
def plot_out_degree_scatter(network):
    """Plots the outgoing degree distribution as scatter plot"""
    degrees_out_dict = network.out_degree()
    degrees_out = np.asarray([ [key,degrees_out_dict[key]] for key in degrees_out_dict])
    degrees_out_unique = np.unique(degrees_out[:,1])
    degrees_out_unique = np.array([v for v in degrees_out_unique if v != 0])
    degrees_out_cleaned = [v for v in degrees_out if v[1] != 0]
    unique, counts = np.unique(degrees_out_cleaned, return_counts=True)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel("In coming Degree")
    plt.scatter(unique, counts)  
    plt.show()
    
def plot_adjacency_matrix(network):
    adjacency = nx.adjacency_matrix(network)
    plt.spy(adjacency, markersize=0.1);
    plt.show()
    
def plot_ranking_component_size(network):
    """Plots a bar chart with the size of the biggest components"""
    comps = list(nx.connected_component_subgraphs(network))
    comps_len = [len(x.nodes()) for x in comps]    
    max_comps_len = sorted(comps_len)[-5:]
    
    plt.bar(list(reversed(range(5))), max_comps_len)
    plt.xlabel("Components")
    plt.ylabel("Number of nodes")
    plt.show()


def analyze_basic(network):
    """Prints out calculated features such as number of nodes and average degree"""
    
    print ("Number of nodes: {}".format(get_number_nodes(network)))
    print ("Number of edges: {}".format(get_number_edges(network)))    
    
    print ("Number of connected components: {}".format(get_number_connected_components(network)))
    
    print ("Number of self loops: {}".format(get_number_self_loops(network)))
    
    print("Size of giant component: {}".format(get_size_giant_component(network)))
    
    print ("Average degree: {}".format(get_avg_degree(network)))
    
    print("Clustering coeeficient: {}".format(get_cluster_coefficient(network)))   

    
    #plot_ranking_component_size(network)

def analyze_degrees(network, directed_network):
    """Prints plots for the degree distribution, directed and undirected"""
    
    print ("Average degree: {}".format(get_avg_degree(network)))
    
    print ("Average in degree: {}".format(get_avg_in_degree(directed_network)))
    print ("Average out degree: {}".format(get_avg_out_degree(directed_network)))
    
    plot_degree_hist(network)
    
    plot_degree_scatter(network)
    
    plot_in_degree_hist(directed_network)
    plot_in_degree_scatter(directed_network)
    
    plot_out_degree_hist(directed_network)
    plot_out_degree_scatter(directed_network)
    
    plot_adjacency_matrix(network)
    
def analyze_ERBA(network):
    """Compares the network to network models"""
    
    print('My network has {} nodes.'.format(len(network.nodes())))
    print('My network has {} edges.'.format(network.size()))
    N = len(network.nodes())
    L = network.size()
    p = 2*L/((N-1)*N)
    er = nx.erdos_renyi_graph(N, p)
    m  = int((L/N)+1)
    ba = nx.barabasi_albert_graph(N, m)
    print('My Erdős–Rényi network has {} nodes.'.format(len(er.nodes())))
    print('My Erdős–Rényi network has {} edges.'.format(er.size()))
    print('My Barabási-Albert network has {} nodes.'.format(len(ba.nodes())))
    print('My Barabási-Albert network has {} edges.'.format(ba.size()))


def get_gamma_power_law(network):
    """Calculate gamma of scale free network"""
    degrees_dict = network.degree()
    degrees = np.asarray([ [key,degrees_dict[key]] for key in degrees_dict])
    degrees_cleaned = [v for v in degrees if v[1] != 0]
    unique, counts = np.unique(degrees_cleaned, return_counts=True)
    
    X = -np.log(unique)
    Y = np.log(counts/len(network.nodes()))
    
    slope, intercept, r_value, p_value, std_err = linregress(Y,X)
    return slope

def get_in_gamma_power_law(network):
    """Calculate gamma for incoming degree of scale free network"""
    degrees_in_dict = network.in_degree()
    degrees_in = np.asarray([ [key,degrees_in_dict[key]] for key in degrees_in_dict])
    degrees_in_cleaned = [v for v in degrees_in if v[1] != 0]
    unique, counts = np.unique(degrees_in_cleaned, return_counts=True)
    
    X = -np.log(unique)
    Y = np.log(counts/len(network.nodes()))
    
    slope, intercept, r_value, p_value, std_err = linregress(Y,X)
    return slope

def get_out_gamma_power_law(network):
    """Calculate gamma for outgoing degree of scale free network"""
    degrees_out_dict = network.out_degree()
    degrees_out = np.asarray([ [key,degrees_out_dict[key]] for key in degrees_out_dict])
    degrees_out_cleaned = [v for v in degrees_out if v[1] != 0]
    unique, counts = np.unique(degrees_out_cleaned, return_counts=True)
    
    X = -np.log(unique)
    Y = np.log(counts/len(network.nodes()))
    
    slope, intercept, r_value, p_value, std_err = linregress(Y,X)
    return slope

def analyze_scale_free(network, directed_network):
    """Calculates gamma for the network and plots a scatter to check gamma"""
    
    # gamma degree
    degrees_dict = network.degree()
    degrees = np.asarray([ [key,degrees_dict[key]] for key in degrees_dict])
    degrees_cleaned = [v for v in degrees if v[1] != 0]
    unique, counts = np.unique(degrees_cleaned, return_counts=True)
    
    slope = get_gamma_power_law(network)
    print ("Gamma: {}".format(slope))

    plt.scatter(unique, counts)
    abline_values = [ (slope - 1)*np.power(i,-slope) for i in unique]
    plt.plot(unique, abline_values, 'b')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    # gamma in degree
    degrees_in_dict = directed_network.in_degree()
    degrees_in = np.asarray([ [key,degrees_in_dict[key]] for key in degrees_in_dict])
    degrees_in_cleaned = [v for v in degrees_in if v[1] != 0]
    unique, counts = np.unique(degrees_in_cleaned, return_counts=True)
    
    slope = get_in_gamma_power_law(directed_network)
    print ("Gamma: {}".format(slope))

    
    plt.scatter(unique, counts)
    abline_values = [ (slope - 1)*np.power(i,-slope) for i in unique]
    plt.plot(unique, abline_values, 'b')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    # gamma out degree
    degrees_out_dict = directed_network.out_degree()
    degrees_out = np.asarray([ [key,degrees_out_dict[key]] for key in degrees_out_dict])
    degrees_out_cleaned = [v for v in degrees_out if v[1] != 0]
    unique, counts = np.unique(degrees_out_cleaned, return_counts=True)
    
    slope = get_out_gamma_power_law(directed_network)
    print ("Gamma: {}".format(slope))
    
    plt.scatter(unique, counts)
    abline_values = [ (slope - 1)*np.power(i,-slope) for i in unique]
    plt.plot(unique, abline_values, 'b')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    
def analyze_clustering(network):
    """Analyses the clustering and separability"""
    
    print("Clustering coeeficient: {}".format(get_cluster_coefficient(network)))   
    
    # took too long
    #laplacian = nx.laplacian_matrix(network)
    #plt.figure(8)
    #plt.spy(laplacian)
    
    #laplacian = sparse.csr_matrix(laplacian).asfptype()
    #plt.figure(9)
    #plt.spy(laplacian)
    #eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian, k=10, which='SA')
    #plt.figure(10)
    #plt.plot(eigenvalues, '.-', markersize=15)
    #plt.show()

def analyze_attribute(items, attr=""):
    """Plots a histgram and a scatter plot for the distribution of a given item"""
    plt.hist(items)
    plt.ylabel("Number of nodes")
    plt.xlabel(attr)
    plt.show()
    
    attrs_unique = np.unique(items)
    attrs_unique = np.array([v for v in attrs_unique if v > 0])
    items_cleaned = [v for v in items if v > 0]
    unique, counts = np.unique(items_cleaned, return_counts=True)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Number of nodes")
    plt.xlabel(attr)
    plt.scatter(unique, counts)
    plt.show()


ms_in_week = 604800000
mintime = 1199145600000

def analyze_attribute_time(network):
    dict = nx.get_edge_attributes(network, "time")
    items = np.array([val-mintime for key,val in dict.items()])/ms_in_week # millis per week
    analyze_attribute(items.tolist(), attr='time')
    
def analyze_attribute_q_votes(network):
    dict = nx.get_edge_attributes(network, "votes_q")
    items = [val for key,val in dict.items()]
    analyze_attribute(items, attr='Question votes')
    
def analyze_attribute_a_votes(network):
    dict = nx.get_edge_attributes(network, "votes_a")
    items = [val for key,val in dict.items()]
    analyze_attribute(items, attr="Answer votes")

def analyze_attributes(network):
    """Prints plots for the distribution of the edge attributes"""
    
    for attr in ['time', 'votes_q', 'votes_a', 'accepted']:
        print (attr)
        dict = nx.get_edge_attributes(network, attr)
        items = [val for key,val in dict.items()]
        analyze_attribute(items, attr=attr)

def analyze_hubs(network, directed_network):
    
    # find five biggest hubs
    degrees_dict = network.degree()
    hub = max(degrees_dict, key=degrees_dict.get)
    print ("Biggest Hub: {}".format(hub))
    
    in_edges = directed_network.in_edges([hub], data=True)
    out_edges = directed_network.out_edges([hub], data=True)
    
    in_times = [edge[2]['time'] for edge in in_edges]
    out_times = [edge[2]['time'] for edge in out_edges] 
    
    plt.hist(in_times)
    plt.show()
    
    plt.hist(out_times)
    plt.show()
    
    
    
