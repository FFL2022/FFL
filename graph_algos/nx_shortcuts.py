'''Networkx Shortcuts'''
import networkx as nx
from collections import Counter, defaultdict


def neighbors_in(u, q):
    ''' Get neighbor having edges into this node
    Parameters
    ----------
    u: str
    q: nx.MultiDiGraph

    Returns
    ----------
    list(str)
    '''
    return list(set(list(u_n for u_n, _ in q.in_edges(u))))


def neighbors_out(u, q):
    return list(set(list(u_n for _, u_n in q.out_edges(u))))


def all_neighbors(u, q):
    uis = neighbors_in(u, q)
    uos = neighbors_out(u, q)
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
