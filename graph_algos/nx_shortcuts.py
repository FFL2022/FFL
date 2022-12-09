'''Networkx Shortcuts'''
import networkx as nx
from collections import Counter, defaultdict
from typing import List, Union
import numpy as np
import inspect


def where_node(*filter_funcs, **kwargs):
    if len(filter_func) == 0 and len(kwargs) == 0:
        return lambda nx_g, x: True
    lambdas = [(lambda nx_g, x: nx_g.nodes[x][k] == v) if not isinstance(v, list) else (lambda nx_g, x: nx_g.nodes[x][k] in v) for k, v in kwargs.items()]
    lambda_f = lambda x: all(f(x) for f in filter_funcs)
    lambda_final = lambda nx_g, x: all(l(nx_g, x) for l in lambdas) and lambda_f(x)
    return lambda_final


def where_node_not(*filter_funcs, **kwargs):
    lambdas = [(lambda nx_g, x: nx_g.nodes[x][k] == v) if not isinstance(v, list) else (lambda nx_g, x: nx_g.nodes[x][k] not in v) for k, v in kwargs.items()]
    lambda_f = lambda x: not any(f(x) for f in filter_funcs)
    lambda_final = lambda nx_g, x: (not any(l(nx_g, x) for l in lambdas)) and lambda_f(x)
    return lambda_final


def where_edge(where_source=None, where_target=None, *filter_funcs, **kwargs):
    if where_source is None:
        where_source = where_node()
    if where_target is None:
        where_target = where_node()
    lambdas_e = [(lambda e: e[k] == v) for k, v in kwargs.items()]
    lambdas_single_filter = lambda u, v, e: all(filter_func(u, v, e) for filter_func in filter_funcs)
    lambdas_multi_filter = lambda u, v, k, e: all(filter_func(u, v, k, e) for filter_func in filter_funcs)
    lambda_final_single = lambda nx_g, u, v, e: (
            where_source(nx_g, u) and
            where_target(nx_g, v) and 
            lambdas_single_filter(u, v, e) and
            all(l(e) for l in lambdas_e)
            )
    lambda_final_multi = lambda nx_g, u, v, k, e: (
            where_source(nx_g, u) and
            where_target(nx_g, v) and 
            lambdas_multi_filter(u, v, k, e) and
            all(l(e) for l in lambdas_e)
            )
    lambda_final = lambda nx_g, x: lambda_final_multi(nx_g, *x) if isinstance(nx_g, nx.MultiGraph) else lambda_final_single(nx_g, *x)
    return lambda_final


def nodes_where(nx_g, *filter_funcs, **kwargs) -> List[Union[str, int]]:
    lambda_final = where_node(*filter_funcs, **kwargs)
    return list([n for n in nx_g.nodes() if lambda_final(nx_g, n)])

def nodes_where_not(nx_g, *filter_funcs, **kwargs):
    lambda_final = where_node(*filter_funcs, **kwargs)
    return list([n for n in nx_g.nodes() if not lambda_final(nx_g, n)])

def update_nodes(nx_g, nodes=None, **kwargs):
    if nodes is None:
        nodes = nx_g.nodes()
    for n in nodes:
        for k, v in kwargs.items():
            nx_g.nodes[n][k] = v

def edges_where(nx_g, where_source, where_target, *filter_funcs, **kwargs):
    l = where_edge(where_source, where_target, *filter_funcs, **kwargs)
    if isinstance(nx_g, nx.MultiGraph):
        return [(u, v, k, e) for u, v, k, e in nx_g.edges(keys=True, data=True)
                if l(nx_g, [u, v, k, e])]
    else:
        return [(u, v, e) for u, v, e in nx_g.edges(data=True) if l(nx_g, (u, v, e))]

def edges_where_not(nx_g, where_source, where_target, *filter_funcs, **kwargs):
    l = where_edge(where_source, where_target, *filter_funcs, **kwargs)
    if isinstance(nx_g, nx.MultiGraph):
        return [(u, v, k, e) for u, v, k, e in nx_g.edges(keys=True, data=True)
                if not l(nx_g, [u, v, k, e])]
    else:
        return [(u, v, e) for u, v, e in nx_g.edges(data=True) if not l(nx_g, (u, v, e))]


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


def combine_multi(vs: List, merge_nodes=False, node2int=None,
                  merge_to_int=True):
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
        if merge_to_int:
            mapping = dict((node_label, i + n_count)
                        for i, node_label in enumerate(
                            sorted(node_labels, key=node2int)))
        else:
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
