import networkx as nx
from graph_algos import nx_shortcuts
import glob
import os
import pickle as pkl
import collections
from tqdm import tqdm


def spanning_tree_filtering(nx_g, target_node):
    queue = collections.deque([target_node])
    visiteds = set()
    while queue:
        n = queue.popleft()
        if n in visiteds:
            continue
        visiteds.add(n)
        queue.extend(nx_shortcuts.all_neighbors(target_node, nx_g)[2])
    return nx_g.subgraph(visiteds)


def combine_to_one_graph(subgraph_dir,
                         kept_props_v=['target_node'],
                         kept_props_e=[]):
    print("Combine to one graph:")
    fps = list(glob.glob(os.path.join(subgraph_dir, "*.pkl")))
    combined_graph = nx.MultiDiGraph()
    for idx in tqdm(range(len(fps))):
        fp = fps[idx]
        nx_g = pkl.load(open(fp, 'rb'))
        if isinstance(nx_g, tuple):
            nx_g = nx_g[0]
        if isinstance(nx_g, nx.DiGraph):
            nx_g_new = nx.MultiDiGraph()
            for n, ndata in nx_g.nodes(data=True):
                nx_g_new.add_node(n, **ndata)
            for u, v, edata in nx_g.edges(data=True):
                nx_g_new.add_edge(u, v, **edata)
            nx_g = nx_g_new
        for n in nx_g.nodes():
            if 'is_target' not in nx_g.nodes[n]:
                nx_g.nodes[n]['is_target'] = 0
        nx_g = spanning_tree_filtering(
            nx_g,
            list([n for n in nx_g if nx_g.nodes[n]['is_target']])[0])
        for n, data in nx_g.nodes(data=True):
            for key in list(data.keys()):
                if key not in kept_props_v:
                    nx_g.nodes[n].pop(key, None)
        for u, v, data in nx_g.edges(data=True):
            for key in list(data.keys()):
                if key not in kept_props_e:
                    data.pop(key, None)

        combined_graph, _ = nx_shortcuts.combine_multi([combined_graph, nx_g],
                                                       node2int=lambda x: x)
    fps_gpickle = list(glob.glob(os.path.join(subgraph_dir, "*.gpickle")))
    for idx in tqdm(range(len(fps_gpickle))):
        fp = fps_gpickle[idx]
        nx_g = nx.read_gpickle(fp)
        if isinstance(nx_g, tuple):
            nx_g = nx_g[0]
        if isinstance(nx_g, nx.DiGraph):
            nx_g_new = nx.MultiDiGraph()
            for n, ndata in nx_g.nodes(data=True):
                nx_g_new.add_node(n, **ndata)
            for u, v, edata in nx_g.edges(data=True):
                nx_g_new.add_edge(u, v, **edata)
            nx_g = nx_g_new
        for n in nx_g.nodes():
            if 'is_target' not in nx_g.nodes[n]:
                nx_g.nodes[n]['is_target'] = 0
        nx_g = spanning_tree_filtering(
            nx_g,
            list([n for n in nx_g if nx_g.nodes[n]['is_target']])[0])
        for n, data in nx_g.nodes(data=True):
            for key in list(data.keys()):
                if key not in kept_props_v:
                    nx_g.nodes[n].pop(key, None)
        for u, v, data in nx_g.edges(data=True):
            for key in list(data.keys()):
                if key not in kept_props_e:
                    data.pop(key, None)

        combined_graph, _ = nx_shortcuts.combine_multi([combined_graph, nx_g],
                                                       node2int=lambda x: x)
    return combined_graph


def check_equal_nx(nx1, nx2, v_attrs=[], e_attrs=[]):
    nx1_nodes = list(sorted(nx1.nodes()))
    nx2_nodes = list(sorted(nx2.nodes()))
    print(nx1.nodes(data=True), nx2.nodes(data=True))
    if len(nx1.nodes()) != len(nx2.nodes()):
        return False
    nx1_attrs = [[] for n in nx1_nodes]
    nx2_attrs = [[] for n in nx2_nodes]
    for attr in v_attrs:
        for i, n in enumerate(nx1.nodes()):
            nx1_attrs[i].append(nx1.nodes[n][attr])
        for i, n in enumerate(nx2.nodes()):
            nx2_attrs[i].append(nx2.nodes[n][attr])
    for i in range(len(nx1_nodes)):
        nx1_attrs[i] = tuple(nx1_attrs[i])
        nx2_attrs[i] = tuple(nx2_attrs[i])
    nx1_is = list(sorted(range(len(nx1_nodes)), key=lambda i: nx1_attrs[i]))
    nx2_is = list(sorted(range(len(nx2_nodes)), key=lambda i: nx2_attrs[i]))
    nx1_nodes = [nx1_nodes[i] for i in nx1_is]
    nx2_nodes = [nx2_nodes[i] for i in nx2_is]
    nx1_attrs = [nx1_attrs[i] for i in nx1_is]
    nx2_attrs = [nx2_attrs[i] for i in nx2_is]
    if tuple(nx1_attrs) != tuple(nx2_attrs):
        return False

    imap1 = {n: i for i, n in enumerate(nx1_nodes)}
    imap2 = {n: i for i, n in enumerate(nx2_nodes)}
    nx1_edges = list(
        sorted(nx1.edges(keys=True, data=True),
               key=lambda e: tuple([imap1[e[0]], imap1[e[1]]] +
                                   [e[3][attr] for attr in e_attrs] + [e[2]])))
    nx2_edges = list(
        sorted(nx2.edges(keys=True, data=True),
               key=lambda e: tuple([imap2[e[0]], imap2[e[1]]] +
                                   [e[3][attr] for attr in e_attrs] + [e[2]])))
    nx1_edges = [(e[0], e[1], e[2]) for e in nx1_edges]
    nx2_edges = [(e[0], e[1], e[2]) for e in nx2_edges]
    if len(nx1_edges) != len(nx2_edges):
        return False
    for attr in e_attrs:
        nx1_attr = tuple(nx1.edges[e][attr] for e in nx1_edges)
        nx2_attr = tuple(nx2.edges[e][attr] for e in nx2_edges)
        if nx1_attr != nx2_attr:
            return False
    return True
