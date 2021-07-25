'''Networkx Shortcuts'''
import networkx as nx
from collections import Counter, defaultdict
from typing import List
import numpy as np


def neighbors_in(u, q, filter_func=None):
    ''' Get neighbor having edges into this node
    Parameters
    ----------
    u: str/int
    q: nx.MultiDiGraph
    filter_func: lambda u, v, k, data -> bool

    Returns
    ----------
    list(str)/list(int) (based on the node id) set in the original graph
    '''
    if filter_func is None:
        return list(set(list(u_n for u_n, _ in q.in_edges(u))))
    else:
        return list(set(list(u_n for u_n, _, k, e in q.in_edges(u, data=True,
                                                                keys=True)
                             if filter_func(u_n, u, k, e))))


def neighbors_out(u, q, filter_func=None):
    ''' Neighbors out (neighbors that this node have at least one edge to)
    Parameters
    ----------
    u: str/int
    q: nx.MultiDiGraph
    filter_func: lambda u, v, k, data -> bool
    Returns
    ----------
    list(str)/list(int) (based on the node id) set in the original graph
    '''
    if filter_func is None:
        return list(set(list(u_n for _, u_n in q.out_edges(u))))
    else:
        return list(set(list(u_n for _, u_n, k, e in q.out_edges(u, data=True,
                                                                 keys=True)
                             if filter_func(u, u_n, k, e))))


def all_neighbors(u, q, filter_func_in=None, filter_func_out=None):
    ''' All neighbors
    Parameters
    ----------
    u: str/int
    q: nx.MultiDiGraph
    filter_func: lambda u, v, k, data -> bool

    Returns
    ----------
    list(str)/list(int) (based on the node id) set in the original graph
    '''
    uis = neighbors_in(u, q, filter_func_in)
    uos = neighbors_out(u, q, filter_func_out)
    return uis, uos, list(set(uis + uos))


def get_in_out_edge_count(v, G):
    ''' Get in and out edge count
    Parameters
    ----------
    v: str
        The node name under inspection
    G: nx.MultiDiGraph
        The graph containing target node to be inspected

    Returns
    ----------
    i_edges_count: dict
        Dictionary counting input edge for each edge type -
        i_edges_count['left']. Edge type organized by 'label' property in
        edge data
    o_edges_count: dict
        Dictionary counting input edge for each edge type -
        o_edges_count['left']. Edge type organized by 'label' property in
        edge data
    '''
    v_out_edges = G.out_edges(v, data=True, keys=True)
    v_in_edges = G.in_edges(v, data=True, keys=True)
    i_edges_count = Counter(list(d['label'] for (i, j, k, d) in v_in_edges))
    o_edges_count = Counter(list(d['label'] for (i, j, k, d) in v_out_edges))
    return i_edges_count, o_edges_count


def maximum_neighbor_degrees(v, G):
    edges_out = G.out_edges(v, keys=True, data=True)
    edges_in = G.out_edges(v, keys=True, data=True)
    neighbor_out = {}
    neighbor_in = {}
    for _, vo, key, edata in edges_out:
        if edata['label'] not in neighbor_out:
            neighbor_out[edata['label']] = G.degree[vo]
        else:
            neighbor_out[edata['label']] = max(G.degree[vo],
                                               neighbor_out[edata['label']])
    for vi, _, key, edata in edges_in:
        if edata['label'] not in neighbor_in:
            neighbor_in[edata['label']] = G.degree[vi]
        else:
            neighbor_in[edata['label']] = max(G.degree[vi],
                                              neighbor_in[edata['label']])
    return neighbor_in, neighbor_out


def combine_multi(vs: List, merge_nodes=False, node2int=None):
    ''' Compose multiple nx graph into 1 graph
    Parameters
    ----------
    merge_nodes:
            Whether to merge nodes of same name or create new one
    Returns
    ----------
    out_nx_g: nx.MultiDiGraph
    batch: [0 0 0 ... 1 1 1 ... N]
    node2int: converting node label to int for comparison
    '''
    if node2int is None:
        if isinstance(vs[0].nodes[0], str):
            def node2int(x): return int(x[1:])
        else:
            def node2int(x): return x
    n_count = 0
    out_nx_g = None
    batch = []
    for v_idx, v in enumerate(vs):
        node_labels = list(v.nodes())
        mapping = dict((node_label, 'n{}'.format(i + n_count))
                       for i, node_label in enumerate(
                           sorted(node_labels, key=node2int)))
        # rename
        new_v = nx.relabel_nodes(v, mapping)
        n_count += len(node_labels)
        batch += [v_idx] * len(node_labels)
        if v_idx == 0:
            out_nx_g = new_v
        else:
            out_nx_g = nx.compose(out_nx_g, new_v)
    return out_nx_g, np.array(batch)
