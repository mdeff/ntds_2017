import networkx as nx
import matplotlib.pyplot as plt


def plot_canton_graph(weights, canton_labels, axi):
    """
    Plots a Graph with the Swiss cantons as nodes, given a set of weights between these cantons. The Plot will also
    highlight if the canton node belongs to the German-speaking, French-speaking or Italian-speaking part of Switzerland
    by plotting the node with a different color.
    :param weights: matrix containing the graph weights between cantons
    :param canton_labels: array containing the labels of the cantons
    :param axi: handle for axis in which to draw
    :return: None
    """

    #plt.figure(3, figsize=(6, 6))

    # Build graph
    graph = nx.Graph(weights)
    pos = nx.spring_layout(graph, scale=1)

    # Construct canton coloring
    french = [6, 7, 10, 12, 22, 23]
    italian = [19]
    german = [c for c in range(26) if c not in french + italian]

    # color canton nodes depending on main language spoken in the canton
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=french,
                           node_color='r',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=italian,
                           node_color='g',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=german,
                           node_color='b',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)

    # color edges
    nx.draw_networkx_edges(graph, pos, width=2.0, alpha=0.5, ax=axi)

    # add labels
    nx.draw_networkx_labels(graph, pos, canton_labels, font_size=16, ax=axi)

    axi.axis('on')
    #axi.savefig("cantons_graph.png")  # save as png
    #axi.show()  # display


def plot_party_graph(weights, party_labels, axi):
    """
    Plots a Graph with the Swiss political parties as nodes, given a set of weights between these parties. The Plot will
    also highlight the connections between a main party and its corresponding youth wing by plotting the connection with
    a thicker line. It will also plot the corresponding nodes with a different color, depending if the party is
    considered to be left/right or liberal/conservative.
    :param weights: matrix containing the graph weights between parties
    :param party_labels: array containing the labels of the parties
    :param axi: handle for axis in which to draw
    :return: None
    """

    #plt.figure(3, figsize=(12, 12))

    graph = nx.Graph(weights)
    pos = nx.spring_layout(graph, scale=1)

    # Construct party coloring according to 2010-14 diagram from Solomon/UZH/Tagesanzeiger April 2014
    left_lib = ['SP', 'GPS', 'EVP', 'PdA']
    left_cons = []
    right_lib = ['glp', 'FDP', 'CVP', 'BDP']
    right_cons = ['SVP', 'EDU', 'Lega']
    inv_party_labels = {v: k for k, v in party_labels.items()}

    # color nodes depending on their left/right or liberal/conservative affiliations
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_labels[p] for p in left_lib],
                           node_color='r',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_labels[p] for p in left_cons],
                           node_color='y',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_labels[p] for p in right_lib],
                           node_color='b',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_labels[p] for p in right_cons],
                           node_color='g',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)

    # color edges differently between pairs of main party and youth wing
    nx.draw_networkx_edges(graph, pos, width=2.0, alpha=0.5, ax=axi)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(17, 18), (7, 8), (13, 14)],
                           width=8, alpha=0.5, edge_color='r', ax=axi)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(1, 2), (4, 5), (11, 12), (9, 10)],
                           width=8, alpha=0.5, edge_color='b',
                           ax=axi)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(15, 16)],
                           width=8, alpha=0.5, edge_color='g',
                           ax=axi)

    # add labels
    nx.draw_networkx_labels(graph, pos, party_labels, font_size=16, ax=axi)

    axi.axis('on')
    #axi.savefig("parties_graph.png")  # save as png
    #axi.show()  # display


def plot_party_graph_voter_and_candidate(weights, party_and_can_labels, axi):
    """
    Plots a Graph with the Swiss political parties as nodes, given a set of weights between these parties. The party
    centroids should contain both the ones from the voter and the ones from the candidate datasets. The Plot will also
    highlight the connections between a main party and its corresponding youth wing by plotting the connection with a
    thicker line. It will also plot the corresponding nodes with a different color, depending if the party is considered
    to be left/right or liberal/conservative.
    :param weights: matrix containing the graph weights between parties
    :param inv_party_and_can_labels: array containing the labels of the parties
    :param axi: handle for axis in which to draw
    :return: None
    """

    #plt.figure(3, figsize=(12, 12))

    graph = nx.Graph(weights)
    pos = nx.spring_layout(graph, scale=1)

    # Construct party coloring according to 2010-14 diagram from Solomon/UZH/Tagesanzeiger April 2014
    left_lib = ['SP', 'GPS', 'EVP', 'PdA']
    left_lib += [l + '_can' for l in left_lib]
    left_cons = []
    right_lib = ['glp', 'FDP', 'CVP', 'BDP']
    right_lib += [l + '_can' for l in right_lib]
    right_cons = ['SVP', 'EDU', 'Lega']
    right_cons += [l + '_can' for l in right_cons]
    inv_party_and_can_labels = {v: k for k, v in party_and_can_labels.items()}


    # color nodes depending on their left/right or liberal/conservative affiliations
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_and_can_labels[p] for p in left_lib],
                           node_color='r',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_and_can_labels[p] for p in left_cons],
                           node_color='y',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_and_can_labels[p] for p in right_lib],
                           node_color='b',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[inv_party_and_can_labels[p] for p in right_cons],
                           node_color='g',
                           node_size=500,
                           alpha=0.8,
                           ax=axi)

    # color edges
    nx.draw_networkx_edges(graph, pos, width=2.0, alpha=0.5, ax=axi)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(17, 18), (7, 8), (13, 14)],
                           width=8, alpha=0.5, edge_color='r', ax=axi)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(1, 2), (4, 5), (11, 12), (9, 10)],
                           width=8, alpha=0.5, edge_color='b', ax=axi)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=[(15, 16)],
                           width=8, alpha=0.5, edge_color='g', ax=axi)

    # add labels
    nx.draw_networkx_labels(graph, pos, party_and_can_labels, font_size=16, ax=axi)

    axi.axis('on')
    #plt.savefig("parties_and_candidates_graph.png")  # save as png
    #plt.show()  # display
