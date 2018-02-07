"""
Various methods related to graph construction and analysis
"""

from preprocess import *

import numpy as np
import networkx as nx
import community
from matplotlib import pyplot as plt


def build_link_exp_decay(adj, weight, words_map, words, from_index, to_index, max_dist, stopwords, links_to_stopwords=True, self_links=False):
    """
    Builds a link from a given word in the graph to another word at a defined index.
    The farther the other word, the smaller the edge between them.
    In this variant, the weight of the edges decays exponentially with the distance
    """
    words_len = len(words)
    links_made = 0
    while to_index < words_len and (words[to_index] in string.punctuation or (not links_to_stopwords and words[to_index] in stopwords) or (not self_links and words[to_index] == words[from_index])):
        to_index += 1
        weight /= 2

    if (to_index - from_index) <= max_dist and to_index < len(words):
        links_made = 1
        adj[words_map[words[from_index]], words_map[words[to_index]]] = adj[words_map[words[from_index]], words_map[words[to_index]]] + weight
    weight /= 2
    return weight, to_index + 1, links_made

def build_link(adj, weight, words_map, words, from_index, to_index, max_dist, stopwords, links_to_stopwords=True, self_links=False):
    """
    Builds a link from a given word in the graph to another word at a defined index.
    The farther the other word, the smaller the edge between them.
    """
    links_made = 0
    if(weight <= 0):
        weight, to_index + 1, links_made
    words_len = len(words)

    while to_index < words_len and (words[to_index] in string.punctuation or (not links_to_stopwords and words[to_index] in stopwords) or (not self_links and words[to_index] == words[from_index])):
        to_index += 1
        weight -= 1

    if (to_index - from_index) <= max_dist and to_index < len(words):
        links_made = 1
        adj[words_map[words[from_index]], words_map[words[to_index]]] = adj[words_map[words[from_index]], words_map[words[to_index]]] + weight
    weight -= 1
    return weight, to_index + 1, links_made

def build_graph(lemmas, lemmas_map, max_dist=20, nlinks=4, max_weight=16, lang=None, links_from_stopwords=True, links_to_stopwords=True, self_links=False):
    len_dist_lemmas = len(lemmas_map)
    len_lemmas = len(lemmas)
    adj = np.zeros((len_dist_lemmas, len_dist_lemmas))
    if(lang != None and (not links_from_stopwords or not links_to_stopwords)):
        stopwords = nltk.corpus.stopwords.words(lang)
    for index, lemma in enumerate(lemmas):
        if lemma in string.punctuation or (not links_from_stopwords and lemma in stopwords):
            continue
        weight = max_dist
        next_index = index + 1
        total_links_made = 0

        for i in range(0, max_dist):
            weight, next_index, links_made = build_link(adj, weight, lemmas_map, lemmas, index, next_index, max_dist, stopwords, links_to_stopwords, self_links)
            total_links_made += links_made

            if(total_links_made >= nlinks or weight <= 0):
                break

    return adj

def text_to_graph(text, undirected=False, subsample=1., normalization="lem", lang="english", words_lower=True, no_punct_nodes=True, nlinks=4, max_dist=20, max_weight=16, ignore_punct=True, ignore_stopwords=False, links_from_stopwords=True, links_to_stopwords=True, self_links=False, return_words_map=False):
    if(ignore_stopwords):
        links_from_stopwords = False
        links_to_stopwords = False

    if normalization == "lem":
        words = words_lems(text, lower=words_lower, ignore_punct=ignore_punct)
    elif normalization == "stem":
        words = words_stems(text, lang=lang, lower=words_lower)

    if(subsample < 1. and subsample > 0):
        sub_len = subsample * len(words)
        words = words[:int(sub_len)]
    words_map = words_to_int(words, lang=lang, ignore_punct=no_punct_nodes, ignore_stopwords=ignore_stopwords)

    graph = build_graph(words, words_map, lang=lang, max_dist=max_dist, max_weight=max_weight, links_from_stopwords=links_from_stopwords, links_to_stopwords=links_to_stopwords)
    if(undirected):
        graph += graph.T

    if(return_words_map):
        return (graph, words_map)
    else:
        return graph

def get_n_closest_words(graph, word_map, word, n_words=10):
    index = word_map[word]
    word_map_inversed = {i[1]:i[0] for i in word_map.items()}
    return [word_map_inversed[np.argsort(graph[index])[::-1][i]] for i in range(n_words)]

def sparsity(m):
    return 1 - np.count_nonzero(m) / m.size

def np_to_nx(M, words_map=None):
    G = nx.from_numpy_matrix(M)
    if(words_map != None):
        words_map_inv = {e[1]:e[0] for e in words_map.items()}
        for n in G:
            G.nodes[n]["word"] = words_map_inv[n]

    return G

def compute_betweenness(G, weight="weight"):
    betweenness = nx.betweenness_centrality(G, weight=weight)
    for n in G:
        G.nodes[n]["betweenness"] = betweenness[n]

    return betweenness

def scale_betweenness(betweenness, min_=10, max_=120):
    """
    Scales the values of the betweenness dictionary to a certain range of values
    """
    max_el = max(betweenness.items(), key=lambda el: el[1])[1]
    mult = max_ / (max_el + min_)
    betweenness_scaled = {k: mult*v + min_ for k,v in betweenness.items()}

    return betweenness_scaled

def community_partition(G, weight="weight"):
    if(weight == "betweenness" and G.nodes()[0].get("betweenness") == None):
        compute_betweenness(G)

    return community.best_partition(G, weight=weight)

def communities(G, draw=True, cmap=None, pos=None, partition=None, betweenness_scaled=None):
    """
    Computes the communities using the Louvain heuristics
    """
    if(partition == None):
        partition = community_partition(G, weight="betweenness")
    if(betweenness_scaled == None):
        if(G.nodes()[0].get("betweenness") == None):
            betweenness = compute_betweenness(G, "betweenness")
        else:
            betweenness = nx.get_node_attributes(G, "betweenness")
        betweenness_scaled = scale_betweenness(betweenness)
    if(pos == None):
        pos = nx.spring_layout(G)

    if(draw and cmap):
        count = 0.
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            sizes = [betweenness_scaled[node] for node in list_nodes]
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size=sizes, node_color=cmap[com])

        nx.draw_networkx_edges(G, pos, alpha=0.05)

    return pos, partition, betweenness_scaled

def induced_graph(original_graph, partition, induced_graph=None, rescale_node_size=1., draw=True, cmap=None, words_map_inv=None, pos=None, betweenness_scaled=None):
    """
    Returns the graph induced from the community partition of the graph
    """
    if(induced_graph == None):
        induced_graph = community.induced_graph(partition, original_graph, weight="weight")

    if(draw and cmap):
        if(pos == None):
            pos = nx.spring_layout(induced_graph)
        w = induced_graph.degree(weight="weight")

        sizes = [w[node]*rescale_node_size for node in induced_graph.nodes()]
        nx.draw(induced_graph, pos=pos, node_size=sizes, node_color=[cmap[n] for n in induced_graph.nodes()])

        labels = {}
        for com in induced_graph.nodes():
            rep = max([nodes for nodes in partition.keys() if partition[nodes] == com], key=lambda n: original_graph.degree(n, weight="weight"))
            labels[com] = words_map_inv[rep]

        nx.draw_networkx_labels(induced_graph, pos, labels, font_size=16)

    return induced_graph
