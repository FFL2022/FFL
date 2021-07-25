from utils.preprocess_helpers import remove_lib
from utils.nx_graph_builder import build_nx_graph_cfg_ast, combine_ast_cfg
from cfg import cfg
import networkx as nx
from graph_algos.cfl_match_general import build_cpi, match_edge, extend_cpi,\
    build_cpi_node_only
from graph_algos.nx_shortcuts import neighbors_out, neighbors_in
import copy

__author__ = "Thanh-Dat Nguyen, thanhdatn@student.unimelb.edu.au"


def ast_node_match_label(n1, ast1, n2, ast2):
    if ast1.node[n1]['ntype'] == ast2.node[n2]['ntype']:
        return True


def ast_node_token_match(n1, ast1, n2, ast2):
    # TODO: consider levenshtein threshold
    return ast1.nodes[n1]['ntype'] == ast2.nodes[n2]['ntype'] and\
        ast1.nodes[n1]['token'] == ast2.nodes[n2]['token']


def simple_top_down_ast_match(ast1: nx.MultiDiGraph, ast2: nx.MultiDiGraph):
    # We don't need fine-grained bottom up like GumTree
    # Since it makes no differences in our case where
    # Whether the node moves or not, we automatically assign
    # Error to parent
    forward_mapping = {}
    backward_mapping = {}
    # BFS
    queue1 = [0]
    while len(queue1) > 0:
        new_forward_mapping = copy.deepcopy(forward_mapping)
        new_backward_mapping = copy.deepcopy(backward_mapping)
        n1 = queue1.pop()
        # 1. Top down adding
        n1_parents = neighbors_in(
            n1, ast1,
            lambda u, v, k, e: u in forward_mapping)
        candidates = []
        for n2 in ast2.nodes():
            if not ast_node_token_match(n1, ast1, n2, ast2):
                continue
            # neighbor consensus
            n2_parents = neighbors_in(
                n2, ast2,
                lambda u, v, k, e:
                any([u in forward_mapping[n1_n] for n1_n in forward_mapping]))
            # Check its relation with all of ast1's neighbor in
            for p1 in n1_parents:
                match = False
                for p2 in n2_parents:
                    if match_edge((p1, n1), ast1, (p2, n2), ast2):
                        match = True
                        break

        if ast_node_match_label(n1, ast1, n2, ast2):
            # Also check if the all the input edges match
            pass
            pass
        # Check if it match with any in queue2
        # If it does, add its children and queue2's children in to queue
    pass


def neighbor_parent_consensus_candidates(n1, ast1, ast2, forward_mapping):
    n1_parents = neighbors_in(
        n1, ast1, lambda u, v, k, e: u in forward_mapping)
    # Get corresponding parents in graph 2
    candidates = []
    for i, p in enumerate(n1_parents):
        for p_c in forward_mapping[p]:
            new_cands = neighbors_out(
                p_c, ast2,
                lambda _, n2, k, e: ast_node_token_match(n1, ast1, n2, ast2) and
                match_edge((p, n1), ast1, (p_c, n2), ast2))
            if i == 0:
                candidates.extend(new_cands)
            else:
                candidates = list(set(candidates).intersection(new_cands))
    return candidates


def full_ast_match(ast1: nx.MultiDiGraph, ast2: nx.MultiDiGraph):
    ''' Careful and slow (guaranteed correct) CFL-based AST match '''
    _, _, last_q = build_cpi(
        ast1, ast2, ast_node_token_match, root_name=0)

    forward_mapping = {}
    backward_mapping = {}
    if last_q is not None:
        forward_mapping = {
            n1: last_q.nodes[n1]['candidates']
            for n1 in last_q.nodes()
            if len(last_q.nodes[n1]['candidates']) > 0
        }
    # Try the procedures of careful slow ast match
    queue = list([n for n in last_q.nodes() if n not in forward_mapping])
    queue = list([n for n in queue if any((n_n in forward_mapping) for
                                          n_n in neighbors_in(n, ast1))])
    list_nodes = list(forward_mapping.keys())
    for node in list(last_q.nodes()):
        if node not in forward_mapping:
            last_q.remove_node(node)
    while len(queue) > 0:
        n1 = queue.pop()
        # 1. Use neighbor consensus
        candidates = neighbor_parent_consensus_candidates(n1, ast1, ast2,
                                                          forward_mapping)
        if len(candidates) == 0:
            continue
        new_temp_subgraph = ast1.subgraph(list_nodes + [n1]).copy()
        if len(candidates) == 1:
            # If this has only one candidate: add it
            q = new_temp_subgraph
            for n in list_nodes:
                q.nodes[n]['candidates'] = last_q.nodes[n]['candidates'][:]
            q.nodes[n1]['candidates'] = candidates
        else:
            # Check if it is isomorphic to ast2
            node_dict, q = build_cpi_node_only(
                new_temp_subgraph, ast2, ast_node_token_match, root_name=0)
            # Note: can be faster by caching previous candidates
        '''
        # 2. Use extend cpi
        new_temp_subgraph = ast1.subgraph(list_nodes + [n1]).copy()
        node_dict, q = extend_cpi(
            last_q,
            new_temp_subgraph, ast2, ast_node_token_match, root_name=0)
        '''
        if all(len(q.nodes[n]['candidates']) > 0 for n in q.nodes()):
            last_q = q
            queue.extend(neighbors_out(n1, ast1))
            list_nodes.append(n1)
            forward_mapping = {
                n1: last_q.nodes[n1]['candidates']
                for n1 in last_q.nodes()
                if len(last_q.nodes[n1]['candidates']) > 0
            }

    forward_mapping = {}
    backward_mapping = {}
    if last_q is not None:
        for n2 in ast2.nodes():
            candidates = [n1 for n1 in last_q.nodes()
                          if n2 in last_q.nodes[n1]['candidates']]
            if len(candidates) > 0:
                backward_mapping[n2] = candidates
        forward_mapping = {n1: last_q.nodes[n1]['candidates']
                           for n1 in last_q.nodes()}

    return forward_mapping, backward_mapping


def careful_slow_ast_match(ast1: nx.MultiDiGraph, ast2: nx.MultiDiGraph):
    ''' Careful and slow (guaranteed correct) CFL-based AST match '''
    queue = [0]
    sub_graph = nx.MultiDiGraph()
    last_q = None
    while len(queue) > 0:
        node = queue.pop()
        new_temp_subgraph = ast1.subgraph(
            list(sub_graph.nodes()) + [node]).copy()
        # Check if it is isomorphic to ast2
        node_dict, edge_dict, q = build_cpi(
            new_temp_subgraph, ast2, ast_node_token_match, root_name=0)
        # Note: can be faster by caching previous candidates
        if all(len(q.nodes[n]['candidates']) > 0 for n in q.nodes()):
            last_q = q
            sub_graph = new_temp_subgraph
            queue.extend(neighbors_out(node, ast1))

    forward_mapping = {}
    backward_mapping = {}
    if last_q is not None:
        for n2 in ast2.nodes():
            candidates = [n1 for n1 in last_q.nodes()
                          if n2 in last_q.nodes[n1]['candidates']]
            if len(candidates) > 0:
                backward_mapping[n2] = candidates
        forward_mapping = {n1: last_q.nodes[n1]['candidates']
                           for n1 in last_q.nodes()}
    return forward_mapping, backward_mapping


def get_bug_localization(file1, file2):
    nline_removed1 = remove_lib(file1)
    graph = cfg.CFG("temp.c")

    with open("temp.c", 'r') as f:
        code = [line for line in f]

    nx_cfg1, nx_ast1, _ = build_nx_graph_cfg_ast(graph, code)

    nline_removed2 = remove_lib(file2)
    graph = cfg.CFG("temp.c")
    with open("temp.c", 'r') as f:
        code = [line for line in f]

    nx_cfg2, nx_ast2, _ = build_nx_graph_cfg_ast(graph, code)
    # 2 Scenarios
    # 1.
    # Take differences betweeen AST and AST

    forward_mapping, backward_mapping = full_ast_match(
        nx_ast1, nx_ast2)

    for n_a1 in nx_ast1.nodes():
        nx_ast1.nodes[n_a1]['status'] = 'k' if n_a1 in forward_mapping else 'd'

    for n_a2 in nx_ast2.nodes():
        if n_a2 in backward_mapping:
            continue
        # Get all parents (either 'next sibling' or 'parent-child'
        kept_parents = neighbors_in(
            n_a2, nx_ast2, lambda u, v, k, e: u in backward_mapping)
        for k_p in kept_parents:
            for b_c in backward_mapping[k_p]:
                nx_ast1.nodes[b_c]['status'] = 'i'
    # Map back to CFG and CFG
    for n_c1 in nx_cfg1.nodes():
        nx_cfg1.nodes[n_c1]['status'] = 'k'
    nx_cfg_ast1 = combine_ast_cfg(nx_ast1, nx_cfg1)
    for n_c1 in nx_cfg_ast1.nodes():
        if nx_cfg_ast1.nodes[n_c1]['graph'] != 'cfg':
            continue
        if len(neighbors_out(n_c1, nx_cfg_ast1, lambda u, v, k, e:
                             nx_cfg_ast1.nodes[v]['status'] != 'k' and
                             nx_cfg_ast1.nodes[v]['graph'] == 'ast')) > 0:
            nx_cfg_ast1.nodes[n_c1]['status'] = 'm'     # Modified
    # 2.
    # Take differences between lines
    # Check back to CFG
    # To be implemented
    return nx_ast1, nx_ast2, nx_cfg1, nx_cfg2, nx_cfg_ast1, nline_removed1
