import DataCleaning as dc
import matplotlib.pyplot as plt
import NetworkAnalysis as na
import networkx as nx
import numpy as np

ms_in_week = 604800000
mintime = 1199145600000

def get_latest_time(network):
    """Calculates the time of the last edge"""
    
    edges = network.edges(data=True)
    times = [edge[2]['time'] for edge in edges]
    return max(times)

def plot_t_n(networks):
    """Plots the number of nodes over time"""
    
    t = list(networks.keys())
    n = [ na.get_number_nodes(nw) for key,nw in networks.items() ]
    
    plt.scatter(t,n)
    plt.xlabel('time')
    plt.ylabel('Number of Nodes')
    plt.show()
    
def plot_t_k_max(networks):
    """Plots the maximal degree over time"""
    
    t = list(networks.keys())
    k_max = [ na.get_max_degree(nw) for key,nw in networks.items() ]
    
    plt.scatter(t,k_max)
    plt.xlabel('time')
    plt.ylabel('Max degree')
    plt.show()
    
def plot_t_k_avg(networks):
    """Plots the average degree over time"""
    
    t = list(networks.keys())
    k_avg = [ na.get_avg_degree(nw) for key,nw in networks.items() ]
    
    plt.scatter(t,k_avg)
    plt.xlabel('time')
    plt.ylabel('Average degree')
    plt.show()
    
def plot_t_c(networks):
    """Plots the cluster coefficient over time"""
    
    t = list(networks.keys())
    c = [ na.get_cluster_coefficient(nw) for key,nw in networks.items() ]
    
    plt.scatter(t,c)
    plt.xlabel('time')
    plt.ylabel('Cluster coeff')
    plt.show()
    
def plot_n_k_max(networks):
    """ Plots the maximal degree over number of nodes"""
    
    n = [ na.get_number_nodes(nw) for key,nw in networks.items() ]
    k_max = [ na.get_max_degree(nw) for key,nw in networks.items() ]
    
    plt.scatter(n,k_max)
    plt.xlabel('Number of nodes')
    plt.ylabel('Max degree')
    plt.show()
    
def plot_n_k_avg(networks):
    """Plots the average degree over number of nodes"""
    
    n = [ na.get_number_nodes(nw) for key,nw in networks.items() ]
    k_avg = [ na.get_avg_degree(nw) for key,nw in networks.items() ]
    
    plt.scatter(n,k_avg)
    plt.xlabel('Number of nodes')
    plt.ylabel('Average degree')
    plt.show()
    
def plot_n_c(networks):
    """Plots the cluster coefficient over number of nodes"""
    
    n = [ na.get_number_nodes(nw) for key,nw in networks.items() ]
    c = [ na.get_cluster_coefficient(nw) for key,nw in networks.items() ]
    
    plt.scatter(n,c)
    plt.xlabel('Number of nodes')
    plt.yscale('symlog')
    plt.xscale('symlog')
    plt.show()
    
def plot_k_avg_k_max(networks):
    """Plots the maximal degree over average degree"""
    
    k_avg = [ na.get_avg_degree(nw) for key,nw in networks.items() ]
    k_max = [ na.get_max_degree(nw) for key,nw in networks.items() ]
    
    plt.scatter(k_avg,k_max)
    plt.xlabel('Average degree')
    plt.ylabel('Max degree')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.show()
    
def analyze_evolution(networks):
    """Plots all statistics for time"""
    
    print ("t vs n")
    plot_t_n(networks)
    
    print ("t vs k max")
    plot_t_k_max(networks)    
    
    print ("t vs k avg")
    plot_t_k_avg(networks)    
    
    print ("t vs c")
    plot_t_c(networks)
    
    print ("n vs k max")
    plot_n_k_max(networks)
    
    print ("n vs k avg")
    plot_n_k_avg(networks)
    
    print ("n vs c")
    plot_n_c(networks)
    
    print ("k avg vs k max")
    plot_k_avg_k_max(networks)



def split_network(network):
    """Creates a dict with networks: time -> network"""
    networks = {}
    
    #latest time from DataFrame
    maxi = get_latest_time(network)
    iter_range = 1+int((maxi-mintime)/ms_in_week)
    
    for i in range(iter_range):
        j = i+1
        maxtime = mintime + j*ms_in_week
        
        #Filtering the Graph by time with Max's fancy filter function
        network_at_time = dc.filter_network_attributes(network, mintime, maxtime, -1, -1, -1, -1, -1)
        networks[maxtime] = network_at_time
    
    return networks

def DegreeDynamics(graph, min_degree):
    """Plots the growth of hubs over time"""
    
    # creating node history dictionary
    node_history_dict = {}

    for node in graph.nodes():
        #Filter Selfloop and parallel edges
        times = [edge[2]['time'] for edge in graph.edges([node], data=True)]
        times = sorted(times)
        if len(times) >= min_degree:
            node_history_dict[node] = times
    
    # prep for node Evolution plot
    dict2 = {}
    for key in node_history_dict:
        mol = len(node_history_dict[key])
        y = 0
        proxi_array = []

        for m in range(mol):
            y = y + 1
            x = node_history_dict[key][m]
            proxi_array.append((x,y))

        dict2[key] =  proxi_array

    # the node Evolution plot
    for it in dict2:
        if dict2[it] == []:
            continue

        s = dict2[it]
        x,y = zip(*s)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("degree")
        plt.xlabel("time")
        plt.scatter(x, y, s=4)
        plt.plot(x,y)
    plt.show()