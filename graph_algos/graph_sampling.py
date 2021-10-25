'''Different sampling techniques on graph'''
import random
import networkx as nx
from execution_engine.legacy_data import default_corpus
from execution_engine.clause import UnaryClause, BinaryClause
from graph_algos.spanning_tree_conversion import level_order_traverse
import numpy as np


class VCGraphSampler(object):
    def sample_textual_feature(sub_g, node, max_text_feats: int = 5):
        ''' sample textual feature to unary clauses
        :param :node: node_label
        :param :sub_g: networkx.Graph
        '''
        # TODO: Randomly sample nodes textual features
        sub_g.nodes[node]['C'] = []
        # Random sample these text's unary clause
        n_text = sub_g.nodes[node]['text']
        max_text_feats = min(len(n_text), max_text_feats)
        chars = random.sample(range(0, len(n_text)), max_text_feats)
        chars = [n_text[c] for c in chars]
        c_idxs = []
        for char in chars:
            if char in default_corpus:
                c_idxs.append('c' + str(default_corpus.index(char)))
        for c_idx in c_idxs:
            sub_g.nodes[node]['C'].append(
                UnaryClause(c_idx, 'True', int(node[1:])))
            # TODO: add False clause for text

    def sample_unary_rel_feature(sub_g, node, map_constraint_rel={
        'top_most': 'bt', 'down_most': 'tb',
        'left_most': 'rl', 'right_most': 'lr'
    }):
        ''' sample unary relation features
        :param :node: node_label
        :param :sub_g: networkx.Graph
        '''
        use_rel_feats = np.random.randint(
            2, size=len(map_constraint_rel.keys())) > 0
        out_edges = sub_g.out_edges(node, data=True)
        for r, (rel, etype) in enumerate(map_constraint_rel.items()):
            cond1s = [
                (u, v, e) for (u, v, e) in out_edges
                if e['label'] != map_constraint_rel[rel]]
            val = False
            if not all(cond1 for cond1 in cond1s):  # Match!
                val = True
            if use_rel_feats[r]:
                sub_g.nodes[node]['C'].append(
                    UnaryClause(rel, val, int(node[1:])))

    def sample_binary_rel_feature(sub_g, node):
        '''sampling binary rel feature
        :param :sub_g: sub_graph under consideration (networkx)
        :param :node: node under consideration
        :return list of binary clause (also applied in place for sub_g)
        '''

        out_edges = list(sub_g.out_edges(node, keys=True, data=True))
        # print("Before: ", out_edges)
        use_edge_var = np.random.randint(2, size=len(out_edges)) > 0
        map_edge_dict = {'lr': 'left', 'rl': 'right', 'tb': 'top',
                         'bt': 'down'}
        max_removal_edge = len(out_edges) - 1
        rm_count = 0
        bin_clauses = []
        removed_edges = []
        for i, (_, v, k, e) in enumerate(out_edges):
            # TODO: Randomly drop edges
            if use_edge_var[i] or rm_count >= max_removal_edge:
                # TODO: Check multiple type of edge between two nodes
                # Get this edge
                e['label'] = map_edge_dict[e['label']]
                e['val'] = 'True'
                bin_clauses.append(
                    BinaryClause(e['label'], int(node[1:]), int(v[1:]))
                )
            else:
                rm_count += 1
                removed_edges.append((node, v, k))
        for u, v, k in removed_edges:
            sub_g.remove_edge(u, v, key=k)
        out_edges = list(sub_g.out_edges(node, data=True))
        '''
        if any([e['label'] in map_edge_dict.keys() for _, v, e in out_edges]):
            print("After: ", out_edges)
        '''

        return bin_clauses

    def sample_vc_graph_from_node_dicts(g, node_dicts, centers):
        '''
        :param :g: networkx graph
        :param :node_dicts: list of node_dict in form of mappings
        Example:
            node_dicts: [{'n0': TreeNode('n0')}]
        '''
        sub_graphs = []
        sub_graph_clauses = []
        for c_idx, node_dict in enumerate(node_dicts):
            # Create a subgraph out of that
            # node_labels = list(node_dict.keys()) ## Using default
            # Using level-order traversal
            tree_nodes = level_order_traverse(
                node_dict['n' + str(centers[c_idx])])
            node_labels = [tree_node.node_label for tree_node in tree_nodes]
            mapping = dict((node_label, 'v{}'.format(i))
                           for i, node_label in enumerate(node_labels))
            inv_mapping = dict(('v{}'.format(i), node_label)
                               for i, node_label in enumerate(node_labels))

            sub_g = g.subgraph(node_labels).copy()
            sub_g = nx.relabel_nodes(sub_g, mapping)
            sub_graph_clauses.append([])
            sub_graphs.append(sub_g)
            for node in sub_g.nodes():
                # TODO: Randomly sampling nodes textual features
                sub_g.nodes[node]['C'] = []
                # Random sampling these text's unary clause
                VCGraphSampler.sample_textual_feature(sub_g, node)
                # Random sample top_most, down_most, left_most, right_most
                VCGraphSampler.sample_unary_rel_feature(sub_g, node)
                sub_graph_clauses[-1].extend(sub_g.nodes[node]['C'])
                # Random sample binary rel
                sub_graph_clauses[-1].extend(
                    VCGraphSampler.sample_binary_rel_feature(sub_g, node))
                # Check original edge degree vs subgraph edge degree
                assert g.degree(inv_mapping[node]) >= sub_g.degree(
                    node), "Assertion failed: degree of subgraph node is " +\
                    "greater than original graph degree"
        return sub_graphs, sub_graph_clauses


class SubDocGraphSampler(object):
    def sample_textual_feature(sub_g, node, max_text_feats: int = 15,
                               word_level=False):
        ''' sample textual feature to unary clauses
        :param :node: node_label
        :param :sub_g: networkx.Graph
        '''
        # TODO: Randomly sample nodes textual features
        sub_g.nodes[node]['C'] = []
        # Random sample these text's characer/word clause
        n_text = sub_g.nodes[node]['text']
        max_text_feats = min(len(n_text), max_text_feats)
        if word_level:
            # First split it
            words = n_text.split()
            words = list(set(words))
            max_text_feats = min(len(words), max_text_feats)
            sub_g.nodes[node]['text'] = ' '.join(words)
        else:
            chars = random.sample(range(0, len(n_text)), max_text_feats)
            chars = list(set([n_text[c] for c in chars]))
            sub_g.nodes[node]['text'] = ''.join(chars)
        return sub_g

    def get_all_edge_labels(G: nx.MultiDiGraph):
        ''' Get all unique edge labels from a graph'''
        edge_lbls = []
        for _, _, _, e in G.edges(data=True, keys=True):
            edge_lbls.append(e['label'])
        return list(set(edge_lbls))

    def sample_binary_rel_feature(sub_g, node):
        '''sampling binary rel feature
        :param :sub_g: sub_graph under consideration (networkx)
        :param :node: node under consideration
        :return list of binary clause (also applied in place for sub_g)
        '''

        out_edges = list(sub_g.out_edges(node, keys=True, data=True))
        # print("Before: ", out_edges)
        use_edge_var = np.random.randint(2, size=len(out_edges)) > 0
        max_removal_edge = len(out_edges) - 1
        rm_count = 0
        removed_edges = []
        for i, (_, v, k, e) in enumerate(out_edges):
            # TODO: Randomly drop edges
            if use_edge_var[i] or rm_count >= max_removal_edge:
                # Get this edge
                e['val'] = 'True'
            else:
                rm_count += 1
                removed_edges.append((node, v, k))
        for u, v, k in removed_edges:
            sub_g.remove_edge(u, v, key=k)
        out_edges = list(sub_g.out_edges(node, data=True))
        return sub_g

    def sample_graph_from_node_dicts(g, node_dicts, centers):
        '''
        :param :g: networkx graph
        :param :node_dicts: list of node_dict in form of mappings
        Example:
            node_dicts: [{'n0': TreeNode('n0')}]
        '''
        sub_graphs = []
        for c_idx, node_dict in enumerate(node_dicts):
            # Create a subgraph out of that
            # node_labels = list(node_dict.keys()) ## Using default
            # Using level-order traversal
            tree_nodes = level_order_traverse(
                node_dict['n' + str(centers[c_idx])])
            node_labels = [tree_node.node_label for tree_node in tree_nodes]
            mapping = dict((node_label, 'n{}'.format(i))
                           for i, node_label in enumerate(node_labels))
            inv_mapping = dict(('n{}'.format(i), node_label)
                               for i, node_label in enumerate(node_labels))
            sub_g = g.subgraph(node_labels).copy()
            sub_g = nx.relabel_nodes(sub_g, mapping)
            sub_graphs.append(sub_g)
            for node in sub_g.nodes():
                # TODO: Randomly sampling nodes textual features
                sub_g.nodes[node]['C'] = []
                # Random sampling these text's unary clause
                SubDocGraphSampler.sample_textual_feature(sub_g, node)
                # Random sample binary rel
                SubDocGraphSampler.sample_binary_rel_feature(sub_g, node)
                # Check original edge degree vs subgraph edge degree
                assert g.degree(inv_mapping[node]) >= sub_g.degree(
                    node), "Assertion failed: degree of subgraph node is " +\
                    "greater than original graph degree"
        return sub_graphs
