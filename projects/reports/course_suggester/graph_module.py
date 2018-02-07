import numpy as np
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import re
import networkx as nx
import itertools
from pygsp import graphs, filters, plotting
plotting.BACKEND = 'matplotlib'

# Argument 1: Dictionary course -> topics
# Return: weight matrix
def old_thomas_compute_weight_matrix(courses_dict):
    # Initialize the weight matrix and the list of the values.
    weight_mat=np.zeros((len(courses_dict),len(courses_dict)))
    values_for_each_course = list(courses_dict.values())
    
    # Create a list containing all the unique values.
    unique_values = []
    for i in range(0,len(values_for_each_course)):
        unique_values.extend(values_for_each_course[i])
    unique_values = list(set(unique_values))
    
    # Loop on the values: Find every courses that have the same value v. Then add a weight between them.
    for i in range(0,len(unique_values)):
        # Variable to store the index of the courses that have the value i in common
        courses_index_with_value_i = []
        for j,lst in enumerate(values_for_each_course):
            for k,value in enumerate(lst):
                if value == unique_values[i]:
                    courses_index_with_value_i.append(j)
        # Add weight between the courses.
        for j in range(0,len(courses_index_with_value_i)):
            for k in range(j+1,len(courses_index_with_value_i)):
                weight_mat[courses_index_with_value_i[j],courses_index_with_value_i[k]] += 1
                weight_mat[courses_index_with_value_i[k],courses_index_with_value_i[j]] += 1        
    return weight_mat

# Argument 1: dictionary topic -> course (topic = StudyPlans or Professors..)
# Argument 2 (optional): dictionary course -> index
# Argument 3 (optional): weight added for one link between two edges. Default = 1
# Return tuple: (weight_matrix) if argument 2 given
#               (weight_matrix, dictionary course -> index) if argument 2 not given

def compute_weight_matrix(*args):
    # Initialize the dictionaries depending on the inputs
    topic_dict = args[0]
    if(len(args) > 1):
        courses_index_dict = args[1]
    else:
        # The index have to be determined
        unique_courses = []
        courses_index_dict = {}
        # Find each courses
        for i in range(0,len(list(topic_dict.values()))):
            unique_courses.extend(list(topic_dict.values())[i])
        unique_courses = list(set(unique_courses))
        # Create dictionary courses -> index
        courses_index_dict=dict(zip(unique_courses, np.arange(len(unique_courses))))
    # Initialize the weight matrix
    weight_mat=np.zeros((len(courses_index_dict),len(courses_index_dict)))
    # Define the weights for each edges
    if(len(args) > 2):
        w = args[2]
    else:
        w = 1      
    # For all topics, find links between courses
    for topic in topic_dict.keys():
        for course1, course2 in itertools.combinations(topic_dict[topic], 2):
            weight_mat[courses_index_dict[course1],courses_index_dict[course2]]+=w
            weight_mat[courses_index_dict[course2],courses_index_dict[course1]]+=w
    # Return the weight matrix and the index dictionary if it was not given as an input.
    if(len(args) > 1):
        return (weight_mat)
    else:
        return (weight_mat, courses_index_dict)
    
def compute_graph(weight_mat):
    # Create the graph: nx -> weight matrix
    Graph = nx.from_numpy_matrix(weight_mat)
    # Initialize the plot and the position of the nodes.
    plt.figure(1,figsize=(16,16)) 
    pos = nx.spring_layout(Graph)
    # Draw and plot the nodes, labels and edges.
    nx.draw_networkx_nodes(Graph, pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(Graph, pos)
    nx.draw_networkx_edges(Graph, pos)
    plt.show()
    return Graph

def compute_graph_pygsp(weight_mat):
    G=graphs.Graph(weight_mat)
    G.compute_laplacian("normalized")
    G.compute_fourier_basis(recompute=True)
    G.set_coordinates(G.U[:,1:3])
    return G