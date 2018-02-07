import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os.path
import networkx as nx


def generate_stubs(g):
    """Generates lists of stubs containing `d` stubs for each node, where `d` is the degree of the node."""
    stubs_array = np.array([], dtype=np.int32)
    # adds num_of_stubs stubs for every node index according to its degree 
    index_degree_pairs = sorted(list(dict(nx.degree(g)).items()), key=lambda x: x[0])
    for ind, num_stubs in index_degree_pairs:
        if num_stubs != 0:
            stubs_array = np.append(stubs_array, ([ind]*num_stubs))
    np.random.shuffle(stubs_array)
    return stubs_array

def get_max_pair(pairs):
    """Returns the index-degree pair, corresponding to the element with at most stubs."""
    pair = sorted(pairs, key=lambda x: x[1], reverse=True)[0]
    if pair[1] == 0:
        return None
    return pair

def greedy_configuration(g):
    """Generates a random graph with degree distribution as close as possible to the graph passed as 
    argument to the function."""
    stubs = generate_stubs(g)
    graph = nx.empty_graph()
    pairs = dict(nx.degree(g)) # index-degree pairs
    highest = get_max_pair(list(pairs.items()))

    # Used to keep up with the number of processed stubs in every moment
    total = sum([p[1] for p in list(pairs.items())])/2 
    processed = 0

    while highest != None:
        source = highest[0] # the node that is the source in this itteration

        # delete the stubs that correspond to the stubs of the source in order to prevent loops
        elem_indices = np.where(stubs == source)
        stubs = np.delete(stubs, elem_indices)

        # break if you have no stubs to connect to except the ones that create self loops
        if len(stubs) == 0:
            print("Breaking in advance to prevent self-loops")
            print("Only stubs of node %d left" % source)
            break

        stubs_left = highest[1]

        while stubs_left != 0: # loop until you use all of the source stubs
            if len(stubs) == 0: # break if no stubs to connect to are left
                print("Breaking while processing to prevent self-loops")
                print("Only stubs of node %d left" % source)
                break

            # choose a random stub connect it to the source and remove it from the list of stubs
            target_index = np.random.choice(len(stubs))
            target = stubs[target_index]

            if graph.has_edge(source, target):
                elem_indices = np.where(stubs == target)
                if len(np.delete(stubs, elem_indices)) == 0:
                    print("Breaking while processing to prevent self-loops")
                    print("Only stubs of node %d and node %s left" % (source, target))
                    pairs[source] = -pairs[source]
                    break
                else:
                    continue
            else:
                graph.add_edge(source, target, weight = np.random.rand())

            stubs = np.delete(stubs, target_index)
            pairs[target] = pairs[target] - 1
            pairs[source] = pairs[source] - 1

            stubs_left = stubs_left - 1

        # Used to keep up with the number of processed stubs in every moment
        processed = processed + highest[1] - stubs_left
        #print("Processed %d / %d" % (processed, total))

        highest = get_max_pair(list(pairs.items()))
    return (graph, pairs)

def generate_user_user_matrix_from_artist_artist_matrix(user_artist_matrix, artist_artist_matrix):
    """Infers user-user connections based on artist similarity and listening counts contained in the user_artist matrix"""
    friend_friend_reconstructed = np.zeros((user_artist_matrix.shape[0],user_artist_matrix.shape[0]))
    
    for i in range(user_artist_matrix.shape[0]):
        # select the row containing artists connected to user i
        user_artists_weights = user_artist_matrix[i]
        # get the indices of artists connected to user i
        non_zero_artist_weights_indices =  list(np.where(user_artists_weights != 0)[0]) 
        # save the position of the user at user_pos_1
        user_pos_1 = i
    
        # loop through all of artists connected to the user at position user_pos_1
        for j, artist_pos in enumerate(non_zero_artist_weights_indices):
            # save the weight of the connection between user at position user_pos_1 and the artist at position artist_pos
            weight_1 = user_artist_matrix[user_pos_1,artist_pos]

            # select the column containing the connections to users for artist at position artist_pos
            artist_to_users = user_artist_matrix[:,artist_pos]
            # get the indices of users connected to artist at artist_pos
            non_zero_user_weights_indices = list(np.where(artist_to_users != 0)[0])
            non_zero_user_weights_indices.remove(user_pos_1)

            # loop through all of the users connected to the artist at artist_pos
            for z, user_pos_2 in enumerate(non_zero_user_weights_indices):
                # save the weight of the connection between user at user_pos_2 and the artist at position artist_pos
                weight_2 = user_artist_matrix[user_pos_2,artist_pos]
                # set the strength of the connection to the minimum of the two weights
                weight = min(weight_1,weight_2)
                # increase the similarity between the users at positions user_pos_1 and user_pos_2 for the strength 
                # of the path between them
                friend_friend_reconstructed[user_pos_1,user_pos_2] = friend_friend_reconstructed[user_pos_1,user_pos_2]  \
                + weight
            
    for i in range(user_artist_matrix.shape[0]):
        # select the row containing artists connected to user i
        user_artists_weights = user_artist_matrix[i]
        # get the indices of artists connected to user i
        non_zero_artist_weights_indices =  list(np.where(user_artists_weights != 0)[0]) 
        # save the position of the user at user_pos_1
        user_pos_1 = i
    
        # loop through all of artists connected to the user at position user_pos_1
        for j, artist_pos in enumerate(non_zero_artist_weights_indices):
            # save the weight of the connection between user at position user_pos_1 and the artist at position artist_pos
            weight_1 = user_artist_matrix[user_pos_1, artist_pos]

            # get the indices for the artists similar to the artist at artist_pos
            similar_artists_indices = np.where(artist_artist_matrix[artist_pos] != 0)[0]

            # loop through all the artist similar to the artist at position artist_pos
            for w, similar_artist_pos in enumerate(similar_artists_indices):
                # save the similarity strength between artist at positions artist_pos and similar_artist_pos
                similarity_strength = artist_artist_matrix[artist_pos, similar_artist_pos]

                # select the column containing the connections to users for artist at position similar_artist_pos
                artist_to_users = user_artist_matrix[:, similar_artist_pos]

                # get the indices of users connected to artist at artist_pos
                non_zero_user_weights_indices = list(np.where(artist_to_users != 0)[0])
                if user_pos_1 in non_zero_user_weights_indices:
                    continue

                users_connected_to_prev = list(np.where(user_artist_matrix[:, artist_pos] != 0)[0])

                # loop through all of the users connected to the artist at similar_artist_pos
                for z, user_pos_2 in enumerate(non_zero_user_weights_indices):
                    if user_pos_2 in users_connected_to_prev:
                        continue
                    # save the weight of the connection between user at user_pos_2 and the artist at similar_artist_pos
                    weight_2 = user_artist_matrix[user_pos_2, similar_artist_pos]
                    # set the strength of the connection to the minimum of the two weights,
                    # rescaled with the similarity strength between the two artists
                    weight = min(weight_1,weight_2)*similarity_strength
                    # increase the similarity between the users at positions user_pos_1 and user_pos_2 for the strength 
                    # of the path between them
                    friend_friend_reconstructed[user_pos_1, user_pos_2] = friend_friend_reconstructed[user_pos_1,user_pos_2] \
                    + weight
        
    return friend_friend_reconstructed

def compare_networks(original, constructed):
    """Compares the two networks in terms of links in the first network that have been detected in the second one"""
    detected = 0
    not_detected = 0
    
    for node_one, node_two, weight in original.edges(data='weight'):
        if constructed.has_edge(node_one, node_two):
            detected = detected + 1
        else:
            not_detected = not_detected + 1
            
    print("The total number of detected links is %d." % detected)
    print("The total number of not detected links is %d." % not_detected)

# Used for testing
def sample_graph_bfs(G, sample_size, source_node):
    """A helper used to sample a graph from a source node, containing a desired number of nodes"""
    visited = set()
    queue = []
    queue.append(source_node)

    while (len(queue) != 0) and (len(visited) < sample_size):
        curr_node = queue.pop(0)
        if curr_node in visited:
            continue
        visited.add(curr_node)
        neighbors = G.neighbors(curr_node)
        queue = queue + list(neighbors)
    return copy.deepcopy(G.subgraph(visited))