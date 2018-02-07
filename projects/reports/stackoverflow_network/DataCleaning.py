import networkx as nx

def filter_network_attributes(network, min_time, max_time, min_q_votes, max_q_votes, min_a_votes, max_a_votes, flag_accepted, directed=False):
    """
    Filters the edges of the given network based on parameters
    time as milliseconds since epoche
    Returns a network that contains only edges between min and max for time and votes
    Ignore bounds by setting parametes to -1
    For flag_accepted = -1 no nodes are filtered, accepted = 1 removes not accepted edges and accepted = 0 filters removes accepted edges
    """
    
    # filter time
    edges = [(start, end, attrs) 
        for start, end, attrs in network.edges(data=True)
            if ( min_time == -1 or attrs['time'] >= min_time) and (max_time == -1 or attrs['time'] <= max_time)]
    
    # filter q_votes
    edges = [(start, end, attrs) 
        for start, end, attrs in edges
            if ( min_q_votes == -1 or attrs['votes_q'] >= min_q_votes) and (max_q_votes == -1 or attrs['votes_q'] <= max_q_votes)]
    
    # filter a_votes
    edges = [(start, end, attrs) 
        for start, end, attrs in edges
            if ( min_a_votes ==-1 or attrs['votes_a'] >= min_a_votes) and ( max_a_votes == -1 or attrs['votes_a'] <= max_a_votes)]
    
    # filter accepted
    edges = [(start, end, attrs) 
        for start, end, attrs in edges
            if flag_accepted == -1 or ((flag_accepted == 1 and attrs['accepted']) or (flag_accepted == 0 and not attrs['accepted']))]
    
    if directed:
        return nx.DiGraph(edges)
    else: 
        return nx.Graph(edges)
    
    
def filter_network_node_degree(network, min_degree, max_degree):  
    """
    Remove nodes out of degree bounds
    """
    network_filtered = network.copy()
    degrees = network_filtered.degree()
    network_filtered.remove_nodes_from((n for n,d in degrees.items()
        if ( not min_degree == -1 and d < min_degree) or ( not max_degree == -1 and d > max_degree)))
    return network_filtered

def filter_network_gc(network):
    """
    Only take the main component of the network
    """
    comps = nx.connected_component_subgraphs(network)
    max_comp = max(comps, key=len)
    return max_comp

def filter_selfloops(network):
    """Remove self loops edges from the grap"""
    network_cleaned = network.copy()
    self_loops = network_cleaned.selfloop_edges()
    network_cleaned.remove_edges_from(self_loops)
    return network_cleaned