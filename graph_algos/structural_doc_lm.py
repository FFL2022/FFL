from __future__ import print_function, unicode_literals, division
import networkx as nx
from graph_algos.cfl_match import TreeNode, graph2spanning_tree
from pathlib import Path


def get_all_leaves(doc_graph):
    leaves = []
    # Find all nodes with degree 1
    for node in doc_graph.nodes():
        degree = doc_graph.degree(node)
        if degree <= 1:
            leaves.append(node)
    return leaves


def process(doc_graph: nx.Graph):
    q = doc_graph
    leaves = get_all_leaves(doc_graph)
    start_points = ['n0'] + leaves
    node_dicts = []
    all_paths = []

    for node in start_points:
        for q_node in q.nodes():
            q.nodes[q_node]['visited'] = False
            q.nodes[q_node]['distance'] = 999999
        q.nodes[node]['visited'] = True
        q.nodes[node]['distance'] = 0
        node_dicts.append({node: TreeNode(node)})
        graph2spanning_tree(q, node, node_dicts[-1])
        all_paths.extend(node_dicts[-1][node].get_all_paths())
    targets = []
    node_path_dicts = {}
    for node in q:
        node_paths = [i for i in range(len(all_paths)) if all_paths[i][-1][0] == node]
        node_path_dicts[node] = node_paths
    return all_paths, node_path_dicts


def process_to_csv(temp_dataset, out_path: Path):
    # TODO: Implement
    pass


# Instead of this, we get all left most, top most
# Then encode towards rightmost, downmost
# And bidir ==> New model, more interpretable
