import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict
from graph_algos.nx_shortcuts import neighbors_out, neighbors_in
import random

def extract_parent_child_probabilistic_rules(nx_gs: List[nx.MultiDiGraph], ctx_level: int = 1) -> Tuple[Dict[Tuple[Tuple[int], Tuple[int]], int], Dict[Tuple[int], int]]:
    """
    Extract all possible extension rules from parent to child
    :param nx_gs: list of networkx graphs
    :return: dictionary of extension rules, dictionary of context counts
    """
    out_dict = defaultdict(int)
    out_ctx_dict = defaultdict(int)
    for nx_g in nx_gs:
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
            out_dict[ctx, (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label'])] += 1
            out_ctx_dict[ctx] += 1
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
            if not prev_siblings:
                out_dict[ctx, (nx_g.nodes[u]['label'], -1, -1)] += 1
                out_dict[ctx, (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label'])] += 1
            else:
                prev_sibling = max(prev_siblings)
                out_dict[ctx, (nx_g.nodes[u]['label'], nx_g.nodes[prev_sibling]['label'], nx_g.edges[u, prev_sibling, 0]['label'])] += 1
                out_dict[ctx, (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label'])] += 1
            out_ctx_dict[ctx] += 1
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
            prev_sibling_key = (nx_g.nodes[u]['label'], -1, -1) if not prev_siblings else (nx_g.nodes[u]['label'], nx_g.nodes[max(prev_siblings)]['label'], nx_g.edges[u, max(prev_siblings), 0]['label'])
            next_sibling_key = (nx_g.nodes[u]['label'], -1, -1) if not next_siblings else (nx_g.nodes[u]['label'], nx_g.nodes[min(next_siblings)]['label'], nx_g.edges[u, min(next_siblings), 0]['label'])
            out_dict[ctx, prev_sibling_key, (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']), next_sibling_key] += 1
    return out_dict, out_ctx_dict
