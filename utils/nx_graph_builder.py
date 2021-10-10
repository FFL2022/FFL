from utils.utils import ConfigClass
from cfg import cfg
from utils.traverse_utils import build_nx_cfg, build_nx_ast_base,\
    build_nx_ast_full
from graph_algos.nx_shortcuts import neighbors_in, neighbors_out
from graph_algos.nx_shortcuts import combine_multi
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
    ''' Combine ast cfg
    Parameters
    ----------
    nx_ast:
          Short description
    nx_cfg:
          Short description
    Returns
    ----------
    param_name: type
          Short description
    '''
    nx_h_g, batch = combine_multi([nx_ast, nx_cfg])
    for node in nx_h_g.nodes():
        if nx_h_g.nodes[node]['graph'] != 'cfg':
            continue
        # only take on-liners
        # Get corresponding lines
        start = nx_h_g.nodes[node]['start_line']
        end = nx_h_g.nodes[node]['end_line']
        if end - start > 0:  # This is a parent node
            continue

        corresponding_ast_nodes = [n for n in nx_h_g.nodes()
                                   if nx_h_g.nodes[n]['graph'] == 'ast' and
                                   nx_h_g.nodes[n]['start_line'] >= start and
                                   nx_h_g.nodes[n]['start_line'] <= end]
        for ast_node in corresponding_ast_nodes:
            nx_h_g.add_edge(node, ast_node, label='corresponding_ast')
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

    for node in nx_cfg.nodes():
        nx_cfg.nodes[node]['graph'] = 'cfg'
    for node in nx_ast.nodes():
        nx_ast.nodes[node]['graph'] = 'ast'
    return nx_cfg, nx_ast, combine_ast_cfg(nx_ast, nx_cfg)



def augment_with_reverse_edge(nx_g, ast_etypes, cfg_etypes):
    for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
        if e['label'] in ast_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] in cfg_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] == 'corresponding_ast':
            nx_g.add_edge(v, u, label='corresponding_cfg')
        elif e['label'] == 'a_pass_test':
            nx_g.add_edge(v, u, label='t_pass_a')
            nx_g.remove_edge(u, v)
        elif e['label'] == 'a_fail_test':
            nx_g.add_edge(v, u, label='t_fail_a')
            nx_g.remove_edge(u, v)
    return nx_g


def augment_with_reverse_edge_cat(nx_g, ast_etypes, cfg_etypes):
    for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
        if e['label'] in ast_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] in cfg_etypes:
            nx_g.add_edge(v, u, label=e['label'] + '_reverse')
        elif e['label'] == 'corresponding_ast':
            nx_g.add_edge(v, u, label='corresponding_cfg')
        elif e['label'] == 'a_pass_test':
            nx_g.add_edge(v, u, label='t_pass_a')
            nx_g.remove_edge(u, v)
        elif e['label'] == 'a_fail_test':
            nx_g.add_edge(v, u, label='t_fail_a')
            nx_g.remove_edge(u, v)
        elif e['label'] == 'c_pass_test':
            nx_g.add_edge(v, u, label='t_pass_c')
        elif e['label'] == 'c_fail_test':
            nx_g.add_edge(v, u, label='t_fail_c')
    return nx_g

# TODO:
# Tobe considered:
# - Syntax error: statc num[3]; in graph 35
