''' Graph transformation tools'''
from __future__ import print_function, unicode_literals
import networkx as nx
import numpy as np


def make_graph_fully_connected(G: nx.MultiDiGraph):
    ''' Create a copy of graph with augmented edge 'no_connect'
    Parameters
    ----------
    G : nx.MultiDiGraph
        The original graph
    Returns
    ----------
    aug_G: nx.MultiDiGraph
        The augmented graph
    '''
    G = G.copy()
    nodes = list(G.nodes())
    node_idx_map = dict([(node, i) for (i, node) in enumerate(G.nodes())])
    N = len(node_idx_map.keys())
    A_no_connect = np.ones((N, N))
    srcs = []
    dsts = []
    for u, v, _, _ in G.edges(keys=True, data=True):
        srcs.append(node_idx_map[u])
        dsts.append(node_idx_map[v])
    srcs = np.array(srcs, dtype=np.int)
    dsts = np.array(dsts, dtype=np.int)
    A_no_connect[srcs, dsts] = 0
    no_edges = np.argwhere(A_no_connect > 0)
    for e in no_edges:
        G.add_edge(nodes[e[0]], nodes[e[1]], label='no_edge')
    return G


def remove_no_edge(G: nx.MultiDiGraph):
    ''' Create a copy of graph with augmented edge 'no_connect'
    Parameters
    ----------
    G : nx.MultiDiGraph
        The augmented graph
    Returns
    ----------
    aug_G: nx.MultiDiGraph
        The stripped graph
    '''
    G = G.copy()
    for u, v, k, e in G.edges(keys=True, data=True):
        if e['label'] == 'no_connect':
            G.remove_edge(u, v, k)
    return G
