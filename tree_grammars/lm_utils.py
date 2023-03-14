import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict, deque
from graph_algos.nx_shortcuts import neighbors_out, neighbors_in
import random
from tree_grammars.utils import TreeGrammarMeta
from miner.gspan_cork_utils import convert_single_graph_attrs_to_int
from networkx.algorithms import isomorphism


def extract_parent_child_probabilistic_rules(nx_gs: List[nx.MultiDiGraph], ctx_level: int = 1) -> Tuple[Dict[Tuple[Tuple[int], Tuple[int]], int], Dict[Tuple[int], int]]:
    """
    Extract all possible extension rules from parent to child
    :param nx_gs: list of networkx graphs
    :return: dictionary of extension rules, dictionary of context counts
    """
    out_dict = defaultdict(int)
    out_ctx_dict = defaultdict(int)
    for nx_g in nx_gs:
        knew_ctx = set()
        for u, v, data in nx_g.edges(data=True):
            # extract ctx
            ctx = [nx_g.nodes[u]['label']]
            ctx_level_cp, has_parent = ctx_level, bool(neighbors_in(u, nx_g))
            while has_parent and ctx_level_cp > 0:
                ctx_node = nx_g.nodes[neighbors_in(u, nx_g)[0]]['label']
                edge_lbl = nx_g.edges[ctx_node, u, 0]['label']
                has_parent = bool(neighbors_in(ctx_node, nx_g))
                ctx.append(edge_lbl)
                ctx.append(ctx_node)
                ctx_level_cp -= 1
            ctx = tuple(ctx)
            out_dict[ctx, (nx_g.nodes[u]['label'], nx_g.nodes[v]
                           ['label'], data['label'])] += 1
            # count each context exactly once, this is important for later grammar construction step
            if ctx not in knew_ctx:
                out_ctx_dict[ctx] += 1
                knew_ctx.add(ctx)
    return out_dict, out_ctx_dict


def extract_bfs_probabilistic_rules(nx_gs: List[nx.MultiDiGraph], ctx_level: int = 1) -> Tuple[Dict[Tuple[Tuple[int], Tuple[int]], int], Dict[Tuple[int], int]]:
    """
    Extract all possible parent-context based breadth-first rules
    :param nx_gs: list of networkx graphs
    :return: dictionary of breadth-first rules, dictionary of context counts
    """
    out_dict = defaultdict(int)
    out_ctx_dict = defaultdict(int)
    for nx_g in nx_gs:
        # assumption: previous edges added is the same as index in the list
        for u, v, data in nx_g.edges(data=True):
            # extract ctx
            ctx = [nx_g.nodes[u]['label']]
            ctx_level_cp, has_parent = ctx_level, bool(neighbors_in(u, nx_g))
            while has_parent and ctx_level_cp > 0:
                ctx_node = nx_g.nodes[neighbors_in(u, nx_g)[0]]['label']
                edge_lbl = nx_g.edges[ctx_node, u, 0]['label']
                has_parent = bool(neighbors_in(ctx_node, nx_g))
                ctx.append(edge_lbl)
                ctx.append(ctx_node)
                ctx_level_cp -= 1
            ctx = tuple(ctx)
            prev_siblings = list(n for n in neighbors_out(u, nx_g) if n < v)
            prev_sibling_key = tuple(sorted(
                (nx_g.nodes[n]['label'], nx_g.edges[u, n, 0]['label']) for n in prev_siblings))
            out_dict[ctx, prev_sibling_key,
                     (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label'])] += 1
            out_ctx_dict[ctx, prev_sibling_key] += 1
    return out_dict, out_ctx_dict


def extract_inorder_probabilistic_rules(nx_gs: List[nx.MultiDiGraph], ctx_level: int = 1) -> Tuple[Dict[Tuple[Tuple[int], Tuple[int]], int], Dict[Tuple[int], int]]:
    """
    Extract all possible parent-context based in-order rules
    :param nx_gs: list of networkx graphs
    :return: dictionary of in-order rules, dictionary of context counts
    """
    out_dict = defaultdict(int)
    out_ctx_dict = defaultdict(int)
    for nx_g in nx_gs:
        # assumption: previous edges added is the same as index in the list
        for u, v, data in nx_g.edges(data=True):
            # extract ctx
            ctx = [nx_g.nodes[u]['label']]
            ctx_level_cp, has_parent = ctx_level, bool(neighbors_in(u, nx_g))
            while has_parent and ctx_level_cp > 0:
                ctx_node = nx_g.nodes[neighbors_in(u, nx_g)[0]]['label']
                edge_lbl = nx_g.edges[ctx_node, u, 0]['label']
                has_parent = bool(neighbors_in(ctx_node, nx_g))
                ctx.append(edge_lbl)
                ctx.append(ctx_node)
                ctx_level_cp -= 1
            ctx = tuple(ctx)
            prev_siblings = list(n for n in neighbors_out(u, nx_g) if n < v)
            next_siblings = list(n for n in neighbors_out(u, nx_g) if n > v)
            prev_sibling_key = (nx_g.nodes[u]['label'], -1, -1) if not prev_siblings else (
                nx_g.nodes[u]['label'], nx_g.nodes[max(prev_siblings)]['label'], nx_g.edges[u, max(prev_siblings), 0]['label'])
            next_sibling_key = (nx_g.nodes[u]['label'], -1, -1) if not next_siblings else (
                nx_g.nodes[u]['label'], nx_g.nodes[min(next_siblings)]['label'], nx_g.edges[u, min(next_siblings), 0]['label'])
            out_dict[ctx, prev_sibling_key, (nx_g.nodes[u]['label'],
                                             nx_g.nodes[v]['label'], data['label']), next_sibling_key] += 1
            out_ctx_dict[ctx, prev_sibling_key, next_sibling_key] += 1
    return out_dict, out_ctx_dict


def cluster_parent_child_rules(rules: Dict[Tuple[Tuple[int], Tuple[int]], int], ctx_counts: Dict[Tuple[int], int]) -> Dict[Tuple[int]] -> List[Tuple[int]]:
    # NOTE: might lose context between sibling
    out_dict = defaultdict(list)
    for ctx in ctx_counts:
        # find all rules that have the same context and count
        ctx_rules = [rule for rule, count in rules.items(
        ) if rule[0] == ctx and count == ctx_counts[ctx]]
        out_dict[ctx] = ctx_rules
    return out_dict


def cluster_breadth_first_rules(breadth_first_rules_dict, out_ctx_dict_breadth_first):
    # NOTE: Drawback: this wont be precise for rules that has in-context change
    # cluster to get the rules that always happens together
    # 1. get ones without any previous sibling first
    out_dict = defaultdict(list)
    for ctx in out_ctx_dict_breadth_first:
        # find all rules that have the same context and count
        ctx_rules = [rule for rule, count in breadth_first_rules_dict.items(
        ) if rule[0] == ctx and count == out_ctx_dict_breadth_first[ctx]]
        out_dict[ctx] = ctx_rules
    return out_dict


def cluster_inorder_rules(inorder_rules_dict, out_ctx_dict_inorder):
    # NOTE: Drawback: this wont be precise for rules that has multiple in-context change
    # cluster to get the rules that always happens together
    # 1. get ones without any previous sibling first
    out_dict = defaultdict(list)
    for ctx in out_ctx_dict_inorder:
        # find all rules that have the same context and count
        ctx_rules = [rule for rule, count in inorder_rules_dict.items(
        ) if rule[0] == ctx and count == out_ctx_dict_inorder[ctx]]
        out_dict[ctx] = ctx_rules
    return out_dict


def complete_tree_pattern(tree_pattern: nx.MultiDiGraph, clustered_rules: Dict[Tuple[int], List[Tuple[int]]]) -> nx.MultiDiGraph:
    """
    Complete the tree pattern with the clustered rules
    :param tree_pattern: tree pattern to be completed
    :param clustered_rules: clustered rules
    :return: completed tree pattern
    """
    tree_pattern = tree_pattern.copy()
    # identify root
    root = [n for n in tree_pattern.nodes if not neighbors_in(n, tree_pattern)]
    assert len(root) == 1
    queue = collections.deque([root[0]])
    # identify root child
    root_child = neighbors_out(root, tree_pattern)
    # check if root label is in the context
    while queue:
        root = queue.popleft()
        if tree_pattern.nodes[root]['label'] in clustered_rules:
            # get the rules
            rules = clustered_rules[tree_pattern.nodes[root]['label']]
            # add the rules to the tree pattern
            for rule in rules:
                if neighbors_out(root, tree_pattern,
                                 lambda u, v, k, e: e['label'] == rule[1][2] and nx_g.nodes[v]['label'] == rule[1][1]):
                    # edge already exists
                    continue
                # add edge
                new_node = max(tree_pattern.nodes) + 1
                tree_pattern.add_node(new_node, label=rule[1][1])
                tree_pattern.add_edge(root, new_node, label=rule[1][2])
        # add children to queue
        queue.extend(neighbors_out(root, tree_pattern))
    return tree_pattern


def get_tree_pattern_text_pattern(tree_pattern: nx.MultiDiGraph, nx_gs: List[nx.MultiDiGraph], grammarMeta: TreeGrammarMeta) -> str:
    """
    Get the text pattern of the tree pattern
    :param tree_pattern: tree pattern
    :param nx_gs: list of nx graphs
    :return: text pattern
    """
    for nx_g in nx_gs:
        converted_nx_g, mapping = convert_single_graph_attrs_to_int(nx_g, grammarMeta.node_attr_names, grammarMeta.edge_attr_names, grammarMeta.node_types, grammarMeta.edge_types)
        # test if subgraph of nx_g is isomorphic to tree_pattern
        # FInd the instance of the tree pattern in the nx graph by isomorphism
        if nx.algorithms.isomorphism.is_isomorphic(converted_nx_g, tree_pattern):
            # found the instance
            break
        else:
            nx_g = None
    if not nx_g:
        raise ValueError("No graph is isomorphic to the tree pattern")
    # 1. find the position of the isomorphic instance
    isomorphic_instance = nx.algorithms.isomorphism.DiGraphMatcher(converted_nx_g, tree_pattern).subgraph_isomorphisms_iter()
    isomorphic_instance = next(isomorphic_instance)
    # 2. get the text pattern
    # 2.0. Find the text of the root
    root = [n for n in tree_pattern.nodes if not neighbors_in(n, tree_pattern)]
    assert len(root) == 1
    root = root[0]
    root_text = nx_g.nodes[isomorphic_instance[root]]['text']
    # 2.1. Find the nodes in the subtree of the tree pattern instance's children that are not in the tree pattern instance
    root_instance = isomorphic_instance[root]
    # Get all of this subtree
    subtree = nx.algorithms.dag.descendants(nx_g, root_instance)
    subtree.add(root_instance)
    # Get the nodes that are not in the tree pattern instance
    subtree_not_in_tree_pattern = subtree - set(isomorphic_instance.values())
    # 2.2. Get the text of the nodes in the subtree that are not in the tree pattern instance
