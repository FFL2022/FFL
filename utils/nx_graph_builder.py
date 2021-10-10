from utils.utils import ConfigClass
from cfg import cfg
from utils.traverse_utils import build_nx_cfg, build_nx_ast_base,\
    build_nx_ast_full
from utils.preprocess_helpers import get_coverage, remove_lib
from codeflaws.data_format import key2bug, key2fix, key2bugfile,\
    key2fixfile, key2test_verdict, get_gcov_file
from graph_algos.nx_shortcuts import combine_multi, neighbors_out
import networkx as nx


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
                                   nx_h_g.nodes[n]['coord_line'] >= start and
                                   nx_h_g.nodes[n]['coord_line'] <= end]
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


def get_coverage_graph_cfg(key: str, nx_cfg, nline_removed):
    nx_cfg_cov = nx_cfg.copy()

    tests_list = key2test_verdict(key)

    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        coverage_map = get_coverage(covfile, nline_removed)
        test_node = nx_cfg_cov.number_of_nodes()
        nx_cfg_cov.add_node(test_node, name=f'test_{i}',
                            ntype='test', graph='test')
        for node in nx_cfg_cov.nodes():
            # Check the line
            if nx_cfg_cov.nodes[node]['graph'] != 'cfg':
                continue
            # Get corresponding lines
            start = nx_cfg_cov.nodes[node]['start_line']
            end = nx_cfg_cov.nodes[node]['end_line']
            if end - start > 0:     # This is a parent node
                continue

            for line in coverage_map:
                if line == start:
                    # The condition of parent node passing is less strict
                    if coverage_map[line] > 0:
                        nx_cfg_cov.add_edge(
                            node, test_node, label='c_pass_test')
                    else:
                        # 2 case, since a common parent line might only have
                        # 1 line
                        nx_cfg_cov.add_edge(
                            node, test_node, label='c_fail_test')
    return nx_cfg_cov


def get_coverage_graph_ast(key, nx_ast_g, nline_removed):
    nx_ast_g = nx_ast_g.copy()

    tests_list = key2test_verdict(key)

    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        # fail/pass = neg in covfile
        coverage_map = get_coverage(covfile, nline_removed)
        test_node = nx_ast_g.number_of_nodes()
        nx_ast_g.add_node(test_node, name=f'test_{i}',
                           ntype='test', graph='test')
        link_type = 'fail' if 'neg' in covfile else 'pass'
        for line in coverage_map:
            if coverage_map[line] <= 0:
                continue
            for a_node in nx_ast_g.nodes:
                if nx_ast_g.nodes[a_node]['graph'] != 'ast':
                    continue
                if nx_ast_g.nodes[a_node]['coord_line'] == line:
                    queue = [a_node]
                    while len(queue) > 0:
                        node = queue.pop()
                        if len(neighbors_out(node, nx_ast_g,
                                lambda u, v, k, e: v == test_node)
                        ) > 0:
                            # Visited
                            continue
                        nx_ast_g.add_edge(
                            node, test_node, label=f'a_{link_type}_test')
                        queue.extend(neighbors_out(
                            node, nx_ast_g,
                            lambda u, v, k, e: nx_ast_g.nodes[v]['graph'] == 'ast'))

    return nx_ast_g

def get_coverage_graph_cfg_ast(key: str, nx_cfg_ast, nline_removed):
    nx_cat = nx_cfg_ast.copy()  # CFG AST Test, CAT

    tests_list = key2test_verdict(key)

    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        coverage_map = get_coverage(covfile, nline_removed)
        t_n = nx_cat.number_of_nodes()
        nx_cat.add_node(t_n, name=f'test_{i}',
                                ntype='test', graph='test')
        link_type = 'fail' if 'neg' in covfile else 'pass'
        for c_n in nx_cat.nodes():
            # Check the line
            if nx_cat.nodes[c_n]['graph'] == 'cfg':
                # Get corresponding lines
                start = nx_cat.nodes[c_n]['start_line']
                end = nx_cat.nodes[c_n]['end_line']
                if end - start > 0:     # This is a parent node
                    continue

                for line in coverage_map:
                    if line == start:
                        # The condition of parent node passing is less strict
                        if coverage_map[line] <= 0:
                            continue
                        nx_cat.add_edge(
                            c_n, t_n, label=f'c_{link_type}_test')
                        queue = neighbors_out(
                            c_n, nx_cat,
                            lambda u, v, k, e: e['label'] =='corresponding_ast')
                        while len(queue) > 0:
                            a_n = queue.pop()
                            if len(neighbors_out(
                                a_n, nx_cat, lambda u, v, k, e: v == t_n)
                            ) > 0:
                                # Visited
                                continue
                            nx_cat.add_edge(
                                a_n, t_n, label=f'a_{link_type}_test')
                            queue.extend(neighbors_out(
                                a_n, nx_cat,
                                lambda u, v, k, e: nx_cat.nodes[v]['graph'] == 'ast'))
    return nx_cat


def build_nx_cfg_coverage_codeflaws(key):
    ''' Build networkx controlflow coverage heterograph codeflaws
    Parameters
    ----------
    cfg_ast_g:  nx.MultiDiGraph
    key:  str - codeflaws dirname
    Returns
    ----------
    param_name: type
          Short description
    '''
    filename = key2fixfile(key)
    nline_removed = remove_lib(filename)
    graph = cfg.CFG("temp.c")

    with open("temp.c", 'r') as f:
        code = [line for line in f]

    nx_cfg, nx_ast, nx_cfg_ast = build_nx_graph_cfg_ast(graph, code)
    nx_cfg_cov = get_coverage_graph_cfg(key, nx_cfg, nline_removed)

    return nx_cfg, nx_ast, nx_cfg_ast, nx_cfg_cov


def build_nx_cfg_ast_coverage_codeflaws(key: str):
    ''' Build networkx controlflow ast coverage heterograph codeflaws
    Parameters
    ----------
    key: str - codeflaws dirname
    Returns
    ----------
    param_name: type
          Short description
    '''

    filename = key2bugfile(key)
    nline_removed = remove_lib(filename)
    graph = cfg.CFG("temp.c")

    with open("temp.c", 'r') as f:
        code = [line for line in f]
    nx_cfg, nx_ast, nx_cfg_ast = build_nx_graph_cfg_ast(graph, code)
    nx_cfg_ast_cov = get_coverage_graph_cfg_ast(key, nx_cfg_ast, nline_removed)

    return nx_cfg, nx_ast, nx_cfg_ast, nx_cfg_ast_cov


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
