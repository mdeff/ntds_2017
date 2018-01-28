import math
import networkx as nx
import numpy as np

def partitions_normalizer(unnormalized_partitions):
    k = {}
    for j in range(len(unnormalized_partitions)):
        k.update({i: j for i in list(unnormalized_partitions[j])}) 

    return k

def partitions_normalizer_reverse(normalized_partitions):
    k = {}
    for user_id, partition_id in normalized_partitions.items():
        
        if partition_id not in k:
            k[partition_id] = [user_id, ]
        else:
            k[partition_id].append(user_id)

    return k

def partitions_to_adj(graph, partitions):
    partitions_adj_matrix = np.zeros((len(list(set(partitions.values()))), len(list(set(partitions.values())))), dtype=np.int32)
    
    partitions_map = {}
    for idx, partition in enumerate(list(set(partitions.values()))):
        partitions_map[idx] = partition

    for source, target in graph.edges():
        partitions_adj_matrix[partitions_map[partitions[source]], partitions_map[partitions[target]]] += 1 
    
    
    
    return partitions_adj_matrix, partitions_map

def find_best_merge(graph, partitions_map):
    p_list = list(graph.nodes())
    
    merge_points = np.zeros((len(p_list), len(p_list)), dtype=np.int32) - (9999 * np.identity(len(p_list)))
    

    for i in p_list:
        for j in p_list:
            
            if i == j: 
                continue
            
            base = {i:i for i in p_list}
            base[j] = i 

            merge_points[i, j] = community.modularity(base, graph)
    

    position = np.argmax(merge_points)
    soon_merged = partitions_map[np.unravel_index(position, (len(set(list(graph.nodes()))),len(set(list(graph.nodes())))))[0]], \
              partitions_map[np.unravel_index(position, (len(set(list(graph.nodes()))),len(set(list(graph.nodes())))))[1]]
    
    return soon_merged

def merge(partitions, A, B):
    partitions_copy =copy.deepcopy(partitions)
    for i in partitions_copy.keys():
        if partitions_copy[i] == A: 
            partitions_copy[i] = B
    return partitions_copy

def smaller(graph, partitions):
    adj_matrix, partitions_map = partions_to_adj(graph, partitions)
    merge_1, merge_2 = find_best_merge(nx.from_numpy_matrix(adj_matrix), partitions_map)
    new_partitions = merge(partitions, merge_1, merge_2)
    return new_partitions

def partitions_equalirium(graph, partitions_1, partitions_2):
    
    while len(list(set(partitions_1.values()))) > len(list(set(partitions_2.values()))):
        partitions_1 = smaller(graph, partitions_1)
    
    return partitions_1, partitions_2

def eq_point(unnormized_partition_1, unnormized_partition_2):
    
    bucket_1 = set()
    bucket_2 = set()
    
    for i in unnormized_partition_1.values():
        bucket_1.update([ (i,j ) for i,j in [element for element in itertools.product(*[i,i])] if i != j])
    for i in unnormized_partition_2.values():
        bucket_2.update([ (i,j ) for i,j in [element for element in itertools.product(*[i,i])] if i != j])
    
    intersection = bucket_1.intersection(bucket_2)
    
    return len(intersection) / math.sqrt(len(bucket_1) * len(bucket_2))

def point_estimate(graph, partitions_1, partitions_2):
    
    p_1, p_2 = partitions_equalirium(G, partitions, new_partitions)
    return eq_point(unnormized_partition_1, unnormized_partition_2)

