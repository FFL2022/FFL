from utils.utils import ConfigClass
from cfg import cfg
from utils.traverse_utils import build_nx_cfg, build_nx_ast_base,\
    build_nx_ast_full
from graph_algos.nx_shortcuts import neighbors_in, neighbors_out,\
        combine_multi, nodes_where, update_nodes
from utils.gumtree_utils import GumtreeWrapper
import networkx as nx
import json
import os


def augment_cfg_with_content(nx_cfg: nx.MultiDiGraph, code: list):
    ''' Augment cfg with text content (In place)
    Parameters
    ----------
    nx_cfg:  nx.MultiDiGraph
    code: list[text]: linenumber-1 -> text
    Returns
    ----------
    nx_cfg: nx.MultiDiGraph
    '''
    for node in nx_cfg.nodes():
        # Only add these line to child node
        nx_cfg.nodes[node]['text'] = code[
            nx_cfg.nodes[node]['start_line'] - 1] \
            if nx_cfg.nodes[node]['start_line'] == \
            nx_cfg.nodes[node]['end_line'] else ''
    return nx_cfg


def combine_ast_cfg(nx_ast, nx_cfg):
    ''' Combine ast cfg by adding each corresponding edge'''
    nx_h_g, batch = combine_multi([nx_ast, nx_cfg])
    for n_cfg in nodes_where(nx_h_g, graph='cfg'):
        s, e = nx_h_g.nodes[n_cfg]['start_line'], nx_h_g.nodes[n_cfg][
            'end_line']
        if e - s > 0:  # This is a parent node
            continue
        corres_n_asts = [
            n for n in nodes_where(nx_h_g, graph='ast')
            if s <= nx_h_g.nodes[n]['start_line'] <= e
        ]
        for n_ast in corres_n_asts:
            nx_h_g.add_edge(n_cfg, n_ast, label='corresponding_ast')
    return nx_h_g


def build_nx_graph_cfg_ast(graph, code: list, full_ast=True):
    ''' Build nx graph cfg ast
    Parameters
    ----------
    graph: cfg.CFG
    code: list[text]
    Returns
    ----------
    cfg_ast: nx.MultiDiGraph
    '''
    graph.make_cfg()
    ast = graph.get_ast()
    nx_cfg, cfg2nx = build_nx_cfg(graph)
    if code is not None:
        nx_cfg = augment_cfg_with_content(nx_cfg, code)

    if full_ast:
        nx_ast, ast2nx = build_nx_ast_full(ast)
    else:
        nx_ast, ast2nx = build_nx_ast_base(ast)

    update_nodes(nx_cfg, graph='cfg')
    update_nodes(nx_ast, graph='ast')
    return nx_cfg, nx_ast, combine_ast_cfg(nx_ast, nx_cfg)


def augment_with_reverse_edge(nx_g, ast_etypes, cfg_etypes):
    for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
        if e['label'] in ast_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] in cfg_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] == 'corresponding_ast':
            nx_g.add_edge(v, u, label='corresponding_cfg')
        elif e['label'] == 'a_pass_t':
            nx_g.add_edge(v, u, label='t_pass_a')
            nx_g.remove_edge(u, v)
        elif e['label'] == 'a_fail_t':
            nx_g.add_edge(v, u, label='t_fail_a')
            nx_g.remove_edge(u, v)
    return nx_g


def augment_with_reverse_edge_cat(nx_g, ast_etypes=None, cfg_etypes=None):
    if ast_etypes is None:
        ast_etypes = set()
        for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
            if nx_g.nodes[u]['graph'] == 'ast' and\
                    nx_g.nodes[v]['graph'] == 'ast':
                ast_etypes.add(e['label'])
    if cfg_etypes is None:
        cfg_etypes = set()
        for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
            if nx_g.nodes[u]['graph'] == 'cfg' and\
                    nx_g.nodes[v]['graph'] == 'cfg':
                cfg_etypes.add(e['label'])

    for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
        if e['label'] in ast_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] in cfg_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] == 'corresponding_ast':
            nx_g.add_edge(v, u, label='corresponding_cfg')
        elif e['label'] == 'a_pass_t':
            nx_g.add_edge(v, u, label='t_pass_a')
            nx_g.remove_edge(u, v)
        elif e['label'] == 'a_fail_t':
            nx_g.add_edge(v, u, label='t_fail_a')
            nx_g.remove_edge(u, v)
        elif e['label'] == 'c_pass_t':
            nx_g.add_edge(v, u, label='t_pass_c')
        elif e['label'] == 'c_fail_t':
            nx_g.add_edge(v, u, label='t_fail_c')
    return nx_g


# TODO:
# Tobe considered:
# - Syntax error: statc num[3]; in graph 35
