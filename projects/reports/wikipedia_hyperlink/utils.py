import wikipedia
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import plotly.graph_objs as go

from sklearn import linear_model


def explore_page(page_title, network, to_explore, inner=False, all_nodes=None):
    """
    This function explores the Wikipedia page who's title is `page_title`.
    :param page_title: title of the Wikipedia page we want to explore
    :param network: dictionary containing the nodes of the graph. If the current page is a real page, we
    add it to this dictionary.
    :param to_explore: Queue of the nodes to explore. We add all the links contained in the current page to this queue.
    :param inner: Boolean. If we're looking for inner links in the network (last step of the scraping), then there is
    no need to explore disambiguation pages or to append the links of the current page to the `to_explore` queue.
    :param all_nodes: This is the set of all the nodes in the network. It is not None only `inner` is True. This is
    useful in order to find the inner links (we take the intersection of the neighbors with the nodes of the network.
    """

    if page_title not in network.keys():
        # then this page has not been explored yet
        try:
            page = wikipedia.page(page_title)  # get the page
            title = page.original_title
            if title not in network.keys():  # check if the original title has already been explored
                if not inner:
                    network[title] = {'links': page.links, 'categories': page.categories, 'url': page.url}
                    for node in page.links:
                        to_explore.append(node)
                else:
                    links = list(set(page.links).intersection(set(all_nodes)))
                    network[title] = {'links': links, 'categories': page.categories, 'url': page.url}
        except wikipedia.DisambiguationError as e:
            if inner:
                # We are only looking for inner links, no need to explore the disambiguation page.
                return
            print('Disambiguation of : {}'.format(page_title))
            links = e.options  # those are the pages listed in the disambiguation page
            for node in links:
                    to_explore.append(node)
        except wikipedia.PageError:
            # page does not exist nothing we can do
            return
        except wikipedia.RedirectError:
            return


def save_obj(obj, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_bag_of_communities(network, partition):
    """
    :param network: dictionary containing for each key (each node/page) a dictionary containing the page categories.
    :param partition: list of the community assignment
    :return: list of dictionaries, one dictionary per community. Each dictionary contains the categories of all the
    nodes of a given community as keys and the number of pages in the community that have this category as values.
    """
    k = len(set(partition))  # number of communities
    bags_of_categories = [{} for _ in range(k)]
    for i, title in enumerate(network.keys()):
        cats = network[title]['categories']
        if type(partition) == list:
            label = partition[i]
        else:
            label = partition[title]
        for c in cats:
            if c in bags_of_categories[label].keys():
                bags_of_categories[label][c] += 1
            else:
                bags_of_categories[label][c] = 1

    return bags_of_categories


def plot_degree_distribution(graph, figsize=(15, 6), title='Degree distribution'):
    """
    Plot the degree distribution of a given NetworkX graph.
    """
    fig, ax = plt.subplots(figsize=figsize)

    d = list(dict(graph.degree()).values())
    sns.distplot(list(d), bins=16, ax=ax)
    ax.set_ylabel('Number of vertices')
    ax.set_xlabel('Degree')
    ax.set_title(title)
    plt.show()


def get_distribution(a):
    """
    Returns the degree distribution of a given NetworkX graph. The returned
    value is an array whose k'th entry is the probability of a node to have
    the degree k.
    """
    if type(a) == nx.classes.graph.Graph:
        probabilities = np.zeros(len(a) + 1)
        for k in nx.adjacency_matrix(a).sum(axis=1):
            probabilities[k] += 1
        probabilities = probabilities / np.sum(probabilities)
        return probabilities


def print_distribution(graph, a=None, b=None, c=None, d=None):
    """
    Plots a graph's degree distribution in natural, semi-log and log-log scales.
    """
    probability_distribution = get_distribution(graph)

    if a is None:
        a = len(probability_distribution)
    if b is None:
        b = len(probability_distribution)
    if c is None:
        c = len(probability_distribution)
    if d is None:
        d = len(probability_distribution)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))

    ax[0, 0].set_title('Degree distribution')
    ax[0, 0].plot(probability_distribution[:a])

    ax[0, 1].set_title('Semi log x degree distribution')
    ax[0, 1].semilogx(probability_distribution[:b])

    ax[1, 0].set_title('Semi log y degree distribution')
    ax[1, 0].semilogy(probability_distribution[:c], 's')

    ax[1, 1].set_title('Log-log degree distribution')
    ax[1, 1].loglog(probability_distribution[:d], 's')

    plt.show()


def print_denoised_degree_distribution(graph, a=None, b=None, c=None, d=None):
    probability_distribution = get_distribution(graph)

    if a is None:
        a = len(probability_distribution)
    if b is None:
        b = len(probability_distribution)
    if c is None:
        c = len(probability_distribution)
    if d is None:
        d = len(probability_distribution)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))

    ax[0, 0].set_title('Degree distribution')
    ax[0, 0].plot(probability_distribution[:a])

    ax[0, 1].set_title('De-noised degree distribution')
    ax[0, 1].plot(probability_distribution[:b])

    ax[1, 0].set_title('Log-log degree distribution')
    ax[1, 0].loglog(probability_distribution[:c], 's')

    ax[1, 1].set_title('Log-log de-noised degree distribution')
    ax[1, 1].loglog(probability_distribution[:d], 's')

    plt.show()


def linear_regression_coefficient(graph, title, limit=None):
    probability_distribution = get_distribution(graph)

    x = np.where(probability_distribution != 0)[0]
    y = probability_distribution[x]

    logx = np.log(x)
    logy = np.log(y)

    if limit is None:
        limit = len(logx)\

    logx = logx[:limit]
    logy = logy[:limit]

    logx = logx.reshape(-1, 1)
    logy = logy.reshape(-1, 1)

    regression = linear_model.LinearRegression()
    regression.fit(logx, logy)

    print('The best linear approximation is y = {0}x + {1}.'.format(regression.coef_, regression.intercept_))

    print('R2 value for the regression : {}'.format(regression.score(logx, logy)))

    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

    ax[0].scatter(logx, logy, color='C0', label='Distribution')
    ax[0].plot(logx, regression.coef_*logx + regression.intercept_, color='C1', label='Linear approximation')
    ax[0].set_title(title)
    ax[0].legend(loc='upper right')

    sns.regplot(logx, logy[:, 0], ax=ax[1])


def build_communities(partition_type, positions, G, community2color):
    edge_trace = go.Scattergl(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        showlegend=False,
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]][positions]
        x1, y1 = G.node[edge[1]][positions]
        edge_trace['x'] += [x0, x1]
        edge_trace['y'] += [y0, y1]

    node_trace = go.Scattergl(
        x=[],
        y=[],
        text=[],
        mode='markers',
        marker=dict(
            color=[],
            size=10,
            opacity=0.5)
    )

    for node in G.nodes():
        x, y = G.node[node][positions]
        node_trace['x'].append(x)
        node_trace['y'].append(y)

    for node in G.nodes:
        node_trace['marker']['color'].append(community2color[int(G.nodes[node][partition_type])])
        node_trace['text'].append(node)

    data = [edge_trace, node_trace]
    return data


def set_layout(title):
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        title='<br>Communities by {}'.format(title),
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            autotick=True,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            autotick=True,
            ticks='',
            showticklabels=False
        ))
    return layout
