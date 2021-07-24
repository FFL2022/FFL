'''General CFL Match, support hetero directed graph'''
from __future__ import print_function, unicode_literals
import networkx as nx
from collections import defaultdict
from graph_algos.spanning_tree_conversion import TreeNode, get_elbl,\
    graph2spanning_trees
from graph_algos.nx_shortcuts import neighbors_in, neighbors_out,\
    all_neighbors, get_in_out_edge_count, maximum_neighbor_degrees
from typing import List, Set, Dict, Tuple, Optional


def get_all_labels(q: nx.MultiDiGraph, G: nx.MultiDiGraph):
    edge_lbls = []
    for _, _, _, e in q.edges(data=True, keys=True):
        edge_lbls.append(e['label'])
    for _, _, _, e in G.edges(data=True, keys=True):
        edge_lbls.append(e['label'])
    return list(set(edge_lbls))


def gen_un_dict(edge_lbls: List[str]):
    un_dict = {'i': {}, 'o': {}}
    for etype in edge_lbls:
        un_dict['i'][etype] = []
        un_dict['o'][etype] = []
    return un_dict


def match_edge(uv_pair: Tuple, q: nx.MultiDiGraph, ij_pair: Tuple,
               G: nx.MultiDiGraph):
    ''' Partial of neighbor consensus in Deep Graph Matching consensus paper'''
    i, j = ij_pair
    u, v = uv_pair
    # Bipartite matching between edges, has to match all edges
    for veid, vedata in q.get_edge_data(u, v).items():
        matched = False
        for deid, dedata in G.get_edge_data(i, j).items():
            cond1 = vedata['label'] == dedata['label']
            if cond1:
                matched = True
                break
        if not matched:
            return False
    return True


def check_neighbor_degree(u, q, v, G, check_label_func):
    u_is, u_os, u_as = all_neighbors(u, q)
    v_is, v_os, v_as = all_neighbors(v, G)
    for u_i in u_is:
        matched = False
        for v_i in v_is:
            if match_edge((u_i, u), q, (v_i, v), G):
                if check_basic_compatible(u_i, q, v_i, G, check_label_func):
                    matched = True
                    break
        if not matched:
            return False
    for u_o in u_os:
        matched = False
        for v_o in v_os:
            if match_edge((u, u_o), q, (v, v_o), G):
                if check_basic_compatible(u_o, q, v_o, G, check_label_func):
                    matched = True
                    break
        if not matched:
            return False
    return True


def check_degrees(u: str, q: nx.MultiDiGraph, v: str, G: nx.MultiDiGraph):
    u_i_count, u_o_count = get_in_out_edge_count(u, q)
    v_i_count, v_o_count = get_in_out_edge_count(v, G)
    for etype in u_i_count:
        if etype not in v_i_count:
            return False
        if v_i_count[etype] < u_i_count[etype]:
            return False
    for etype in u_o_count:
        if etype not in v_o_count:
            return False
        if v_o_count[etype] < u_o_count[etype]:
            return False
    return True


def check_basic_compatible(u, q, v, G, check_label_func=lambda u, q, v, G:
                           q.nodes[u]['label'] == G.nodes[v]['label']):
    return check_degrees(u, q, v, G) and check_label_func(u, q, v, G)


def cand_verify(u: str, q: nx.MultiDiGraph, v: str, G: nx.MultiDiGraph,
                check_label_func):
    dvn_in, dvn_out = maximum_neighbor_degrees(v, G)
    dun_in, dun_out = maximum_neighbor_degrees(u, q)
    for etype in dun_in:
        if etype not in dvn_in:
            return False
        if dvn_in[etype] < dun_in[etype]:
            return False
    for etype in dun_out:
        if etype not in dvn_out:
            return False
        if dvn_out[etype] < dun_out[etype]:
            return False
    # return True
    return check_neighbor_degree(u, q, v, G, check_label_func)


def check_edge_in(u, u_n, q, G, cnt_dict, edge_type, check_label_func):
    for v_u_n in q.nodes[u_n]['candidates']:
        for _, v, edata in G.out_edges(v_u_n, data=True):
            if not (match_edge((u_n, u), q, (v_u_n, v), G)
                    and check_basic_compatible(u, q, v, G, check_label_func)
                    ):
                continue
            if G.nodes[v]['cnt']['i'][edge_type] == cnt_dict['i'][edge_type]:
                G.nodes[v]['cnt']['i'][edge_type] += 1


def check_edge_out(u, u_n, q, G, cnt_dict, edge_type, check_label_func):
    for v_u_n in q.nodes[u_n]['candidates']:
        for v, _, edata in G.in_edges(v_u_n, data=True):
            if not (match_edge((u, u_n), q, (v, v_u_n), G)
                    and check_basic_compatible(u, q, v, G, check_label_func)
                    ):
                continue
            if G.nodes[v]['cnt']['o'][edge_type] == cnt_dict['o'][edge_type]:
                G.nodes[v]['cnt']['o'][edge_type] += 1


def create_empty_cnt_dict(edge_labels):
    cnt_dict = {'i': {}, 'o': {}}
    for etype in edge_labels:
        cnt_dict['i'][etype] = 0
        cnt_dict['o'][etype] = 0
    return cnt_dict


def gen_edge_dict(edge_labels):
    edge_dict = {'i': {}, 'o': {}}
    for etype in edge_labels:
        edge_dict['i'][etype] = {}
        edge_dict['o'][etype] = {}
    return edge_dict


def get_params_cnt(is_in, u, u_n, q):
    if is_in:
        io_switch = 'i'
        etype = get_elbl((u_n, u), q)
        check_edge_func = check_edge_in
    else:
        io_switch = 'o'
        etype = get_elbl((u, u_n), q)
        check_edge_func = check_edge_out
    return io_switch, etype, check_edge_func

# build cpi


def check_snte(u, u_n, q):
    return q.nodes[u_n]['distance'] == q.nodes[u]['distance']


def check_cnt_match(cnt_dict1, cnt_dict2):
    for etype in cnt_dict1['i'].keys():
        if cnt_dict1['i'][etype] != cnt_dict2['i'][etype]:
            return False
    for etype in cnt_dict1['o'].keys():
        if cnt_dict1['o'][etype] != cnt_dict2['o'][etype]:
            return False
    return True


def compare_level(u, u_n, q):
    lev_u_n = q.nodes[u_n]['distance']
    lev_u = q.nodes[u]['distance']
    if lev_u < lev_u_n:
        return -1
    elif lev_u > lev_u_n:
        return 1
    else:
        return 0


def cpi_top_down(q: nx.MultiDiGraph, G: nx.MultiDiGraph,
                 root: str, node_dict: Dict[str, TreeNode],
                 check_label_func=lambda u, q, v, G:
                 q.nodes['u']['label'] == G.nodes['v']['label']):
    ''' Building CPI top-down
    Parameters
    ----------
    q : nx.MultiDiGraph
        The source graph to be used for matching
    G : nx.MultiDiGraph
        The target graph to be matched against
    root: str
        The anchored chosen root node, can be random if needed
    node_dict : dict
                Dictionary mapping from node name to TreeNode in BFS Tree
    check_label_func: function to check compatible nodes
    Returns
    -------
    int
        Description of anonymous integer return value.
    '''
    r_i_count, r_o_count = get_in_out_edge_count(root, q)
    edge_labels = get_all_labels(q, G)
    for node in node_dict:
        q.nodes[node]['candidates'] = []
        q.nodes[node]['UN'] = gen_un_dict(edge_labels)

    for v in G.nodes():
        # Check degree
        v_i_count, v_o_count = get_in_out_edge_count(v, G)
        if not (check_basic_compatible(root, q, v, G, check_label_func) and
                cand_verify(root, q, v, G, check_label_func)):
            continue
        q.nodes[root]['candidates'].append(v)

    for node in q.nodes():
        q.nodes[node]['visited'] = False

    q.nodes[root]['visited'] = True

    for node in G.nodes():
        G.nodes[node]['cnt'] = create_empty_cnt_dict(edge_labels)

    # TODO: use queue and delimiter instead
    l2n_map = defaultdict(list)
    for node in node_dict:
        lev = q.nodes[node]['distance']
        l2n_map[lev].append(node)

    n_levels = len(list(l2n_map.keys()))
    edge_dict = {}
    for v in G.nodes():
        edge_dict[v] = gen_edge_dict(edge_labels)

    for lev in range(1, n_levels):
        # Forward candidate generation
        for u in l2n_map[lev]:
            u_i_count, u_o_count = get_in_out_edge_count(u, q)
            cnt_dict = create_empty_cnt_dict(edge_labels)
            u_is, u_os, u_as = all_neighbors(u, q)
            for u_n in u_as:
                io_switch, etype, check_edge_func = get_params_cnt(u_n in u_is,
                                                                   u, u_n, q)
                if not q.nodes[u_n]['visited'] and check_snte(u, u_n, q):
                    q.nodes[u]['UN'][io_switch][etype].append(u_n)
                elif q.nodes[u_n]['visited']:
                    check_edge_func(u, u_n, q, G, cnt_dict, etype,
                                    check_label_func)
                    cnt_dict[io_switch][etype] += 1
            for v in G.nodes():
                cnt_match = check_cnt_match(G.nodes[v]['cnt'], cnt_dict)
                # If every count, every time we check u from all other node's
                # position so far, v match, then v is one potential candidates
                if cnt_match and cand_verify(u, q, v, G, check_label_func):
                    q.nodes[u]['candidates'].append(v)
            q.nodes[u]['visited'] = True
            for node in G.nodes():
                G.nodes[node]['cnt'] = create_empty_cnt_dict(edge_labels)
        # Backward candidate pruning
        for u in l2n_map[lev][::-1]:
            cnt_dict = create_empty_cnt_dict(edge_labels)
            for io_switch in q.nodes[u]['UN']:
                check_edge_func = check_edge_in if io_switch == 'i' else check_edge_out
                for etype in q.nodes[u]['UN'][io_switch]:
                    for u_n in q.nodes[u]['UN'][io_switch][etype]:
                        check_edge_func(u, u_n, q, G, cnt_dict, etype,
                                        check_label_func)
                        cnt_dict[io_switch][etype] += 1
            for v in list(q.nodes[u]['candidates'])[:]:
                if not check_cnt_match(G.nodes[v]['cnt'], cnt_dict):
                    q.nodes[u]['candidates'].remove(v)
            # Reset again
            for node in G.nodes():
                G.nodes[node]['cnt'] = create_empty_cnt_dict(edge_labels)
        for u in l2n_map[lev]:
            u_p = node_dict[u].parent.node_label
            uis, uos, uas = all_neighbors(u, q)
            io_switch, etype, _ = get_params_cnt(u_p in uis, u, u_p, q)
            if io_switch == 'i':                    # Tricky: inverse
                neighbors_func = neighbors_out
            else:
                neighbors_func = neighbors_in
            for v_p in q.nodes[u_p]['candidates']:
                for v in list(set(neighbors_func(v_p, G)).intersection(
                        set(q.nodes[u]['candidates']))):
                    if u not in edge_dict[v][io_switch][etype]:
                        edge_dict[v][io_switch][etype] = {u: {u_p: []}}
                    edge_dict[v][io_switch][etype][u][u_p].append(v_p)
    return node_dict, edge_dict, q


def cpi_bottom_up(q: nx.MultiGraph, G: nx.MultiDiGraph,
                  node_dict: Dict[str, TreeNode],
                  edge_dict,
                  check_label_func):
    ''' CPI bottom up candidate pruning
      Parameters
      ----------
      edge_dict: dict
          dictionary that map from (candidate name: node in target graph,
                                    edge_direction('i', 'o'),
                                    etype(label),
                                    q_source_node_name: source node in query),
                                    q_target_node_name: target node in query),
                                    to a list of
      Returns
      ----------
      param_name: type
            Short description
    '''
    edge_labels = get_all_labels(q, G)
    l2n_map = defaultdict(list)
    for node in node_dict:
        lev = q.nodes[node]['distance']
        l2n_map[lev].append(node)

    l2n_map = dict((lev, tuple(node_list))
                   for lev, node_list in l2n_map.items())
    n_levels = len(list(l2n_map.keys()))
    for lev in list(range(n_levels))[::-1]:
        for u in l2n_map[lev]:
            cnt_dict = create_empty_cnt_dict(edge_labels)
            u_is, u_os, u_as = all_neighbors(u, q)
            u_ns = [u_n for u_n in u_as if compare_level(u, u_n, q) > 0]
            for u_n in u_ns:
                io_switch, etype, check_edge_func = get_params_cnt(u_n in u_is,
                                                                   u, u_n, q)
                check_edge_func(u, u_n, q, G, cnt_dict, etype,
                                check_label_func)
                cnt_dict[io_switch][etype] += 1

            for v in q.nodes[u]['candidates'][:]:
                if not check_cnt_match(G.nodes[v]['cnt'], cnt_dict):
                    q.nodes[u]['candidates'].remove(v)
                    # Also remove all adj list
                    for etype in edge_labels:
                        edge_dict[v]['i'][etype].pop(u, None)
                        edge_dict[v]['o'][etype].pop(u, None)
            for v in G.nodes():
                G.nodes[v]['cnt'] = create_empty_cnt_dict(edge_labels)
            for v in q.nodes[u]['candidates'][:]:
                # Final minor heuristic check from the backward directions
                for child_node in node_dict[u].children:
                    u_child = child_node.node_label
                    for etype in edge_labels:
                        if u_child in edge_dict[v]['i'][etype]:
                            if u in edge_dict[v]['i'][etype][u_child]:
                                for maybe_v_child in edge_dict[v]['i'][etype][u_child][u][:]:
                                    if maybe_v_child not in q.nodes[u_child]['candidates'][:]:
                                        edge_dict[v]['i'][etype][u_child][u][:].remove(
                                            maybe_v_child)
                        else:
                            if u_child in edge_dict[v]['o'][etype]:
                                if u in edge_dict[v]['o'][etype][u_child]:
                                    for maybe_v_child in edge_dict[v]['o'][etype][u_child][u][:]:
                                        if maybe_v_child not in q.nodes[u_child]['candidates'][:]:
                                            edge_dict[v]['o'][etype][u_child][u][:].remove(
                                                maybe_v_child)
    return node_dict, edge_dict, q


def extend_cpi_top_down(last_q: nx.MultiDiGraph, q: nx.MultiDiGraph,
                        G: nx.MultiDiGraph,
                        root: str, node_dict: Dict[str, TreeNode],
                        check_label_func=lambda u, q, v, G:
                        q.nodes['u']['label'] == G.nodes['v']['label']):
    ''' Extending CPI top-down with new node, useful for mining
    TODO: implement
    Parameters
    ----------
    q : nx.MultiDiGraph
        The source graph to be used for matching,
        has some overlapping with last_q
    G : nx.MultiDiGraph
        The target graph to be matched against
    root: str
        The anchored chosen root node, can be random if needed
    node_dict : dict
                Dictionary mapping from node name to TreeNode in BFS Tree
    check_label_func: function to check compatible nodes
    Returns
    -------
    int
        Description of anonymous integer return value.
    '''
    r_i_count, r_o_count = get_in_out_edge_count(root, q)
    edge_labels = get_all_labels(q, G)
    for node in node_dict:
        q.nodes[node]['candidates'] = []
        q.nodes[node]['UN'] = gen_un_dict(edge_labels)

    for v in G.nodes():
        # Check degree
        v_i_count, v_o_count = get_in_out_edge_count(v, G)
        if not (check_basic_compatible(root, q, v, G, check_label_func) and
                cand_verify(root, q, v, G, check_label_func)):
            continue
        q.nodes[root]['candidates'].append(v)

    for node in q.nodes():
        q.nodes[node]['visited'] = False

    q.nodes[root]['visited'] = True

    for node in G.nodes():
        G.nodes[node]['cnt'] = create_empty_cnt_dict(edge_labels)

    # TODO: use queue and delimiter instead
    l2n_map = defaultdict(list)
    for node in node_dict:
        lev = q.nodes[node]['distance']
        l2n_map[lev].append(node)

    n_levels = len(list(l2n_map.keys()))
    edge_dict = {}
    for v in G.nodes():
        edge_dict[v] = gen_edge_dict(edge_labels)

    for lev in range(1, n_levels):
        # Forward candidate generation
        for u in l2n_map[lev]:
            u_i_count, u_o_count = get_in_out_edge_count(u, q)
            cnt_dict = create_empty_cnt_dict(edge_labels)
            u_is, u_os, u_as = all_neighbors(u, q)
            for u_n in u_as:
                io_switch, etype, check_edge_func = get_params_cnt(u_n in u_is,
                                                                   u, u_n, q)
                if not q.nodes[u_n]['visited'] and check_snte(u, u_n, q):
                    q.nodes[u]['UN'][io_switch][etype].append(u_n)
                elif q.nodes[u_n]['visited']:
                    check_edge_func(u, u_n, q, G, cnt_dict, etype,
                                    check_label_func)
                    cnt_dict[io_switch][etype] += 1
            for v in G.nodes():
                cnt_match = check_cnt_match(G.nodes[v]['cnt'], cnt_dict)
                # If every count, every time we check u from all other node's
                # position so far, v match, then v is one potential candidates
                if cnt_match and cand_verify(u, q, v, G, check_label_func):
                    q.nodes[u]['candidates'].append(v)
            q.nodes[u]['visited'] = True
            for node in G.nodes():
                G.nodes[node]['cnt'] = create_empty_cnt_dict(edge_labels)
        # Backward candidate pruning
        for u in l2n_map[lev][::-1]:
            cnt_dict = create_empty_cnt_dict(edge_labels)
            for io_switch in q.nodes[u]['UN']:
                check_edge_func = check_edge_in if io_switch == 'i' else check_edge_out
                for etype in q.nodes[u]['UN'][io_switch]:
                    for u_n in q.nodes[u]['UN'][io_switch][etype]:
                        check_edge_func(u, u_n, q, G, cnt_dict, etype,
                                        check_label_func)
                        cnt_dict[io_switch][etype] += 1
            for v in list(q.nodes[u]['candidates'])[:]:
                if not check_cnt_match(G.nodes[v]['cnt'], cnt_dict):
                    q.nodes[u]['candidates'].remove(v)
            # Reset again
            for node in G.nodes():
                G.nodes[node]['cnt'] = create_empty_cnt_dict(edge_labels)
        for u in l2n_map[lev]:
            u_p = node_dict[u].parent.node_label
            uis, uos, uas = all_neighbors(u, q)
            io_switch, etype, _ = get_params_cnt(u_p in uis, u, u_p, q)
            if io_switch == 'i':                    # Tricky: inverse
                neighbors_func = neighbors_out
            else:
                neighbors_func = neighbors_in
            for v_p in q.nodes[u_p]['candidates']:
                for v in list(set(neighbors_func(v_p, G)).intersection(
                        set(q.nodes[u]['candidates']))):
                    if u not in edge_dict[v][io_switch][etype]:
                        edge_dict[v][io_switch][etype] = {u: {u_p: []}}
                    edge_dict[v][io_switch][etype][u][u_p].append(v_p)
    return node_dict, edge_dict, q



def build_cpi(q, G, check_label_func, root_name='n0'):
    q = q.copy()
    G = G.copy()
    roots, node_dicts = graph2spanning_trees(q, root_name)
    edge_dicts = []
    for root, node_dict in zip(roots, node_dicts):
        node_dict, edge_dict, q = cpi_top_down(q, G, root.node_label,
                                               node_dict, check_label_func)
        node_dict, edge_dict, q = cpi_bottom_up(q, G, node_dict, edge_dict,
                                                check_label_func)
        edge_dicts.append(edge_dict)
    return node_dicts, edge_dicts, q
