'''Tests to ensure that the general CFL match is correct'''
from __future__ import print_function, unicode_literals
from execution_engine.temp_data_loader import TempDataset
from execution_engine.legacy_data import default_input_features
from graph_algos.cfl_match_general import build_cpi
from graph_algos.spanning_tree_conversion import sample_bfs_from_graph
import time
from graph_algos.graph_sampling import SubDocGraphSampler
import random
import networkx as nx


def check_text_contain_char(u: str, q: nx.MultiDiGraph,
                            v: str, G: nx.MultiDiGraph):
    ''' Check text contain char
    Parameters
    ----------
    u:  str
        Node in the query graph
    q:  nx.MultiDigraph
        Query graph
    v:  str
        Node in the target graph
    G:  nx.MultiDiGraph
        Target graph
    Returns
    ----------
    output: boolean
          Check if two text match or not under char condition
    '''
    txt_src = q.nodes[u]['text']
    txt_tgt = G.nodes[v]['text']
    return all(c in txt_tgt for c in txt_src)


def check_text_contain_words(u: str, q: nx.MultiDiGraph,
                             v: str, G: nx.MultiDiGraph):
    ''' Check text contain words
    Parameters
    ----------
    u:  str
          Node in the query graph
    q:  nx.MultiDigraph
          query graph
    v:  str
          Node in the target graph
    G:  nx.MultiDiGraph
          target graph
    Returns
    ----------
    output: boolean
          Check if two text match or not under word condition
    '''
    words_src = q.nodes[u]['text'].split()
    words_tgt = G.nodes[v]['text'].split()
    return all(w in words_tgt for w in words_src)


if __name__ == '__main__':
    temp_dataset = TempDataset(default_input_features)
    for gid, doc_graph in enumerate(temp_dataset):
        max_depth = random.randint(1, 10)
        g = doc_graph.g
        node_dicts, centers = sample_bfs_from_graph(g, 15, max_depth)
        sub_graphs = SubDocGraphSampler.sample_graph_from_node_dicts(
            g, node_dicts, centers)

        for i, sub_graph in enumerate(sub_graphs):
            start = time.time()
            node_dict, edge_dict, q = build_cpi(sub_graph, g,
                                                check_text_contain_char, 'n0')
            end = time.time()
            print(centers[i], q.nodes['n0']['candidates'], 'in ', end-start)
