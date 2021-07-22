from utils.utils import ConfigClass
from cfg import cfg
from utils.traverse_utils import build_nx_cfg, build_nx_ast
from utils.preprocess_helpers import get_coverage, remove_lib
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


def build_nx_graph_cfg_ast(graph, code: list):
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

    nx_ast, ast2nx = build_nx_ast(ast)

    for node in nx_cfg.nodes():
        nx_cfg.nodes[node]['graph'] = 'cfg'
    for node in nx_ast.nodes():
        nx_ast.nodes[node]['graph'] = 'ast'

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
    return nx_cfg, nx_ast, nx_h_g


def build_nx_cfg_coverage_codeflaws(data_codeflaws: dict):
    ''' Build networkx controlflow coverage heterograph codeflaws
    Parameters
    ----------
    cfg_ast_g:  nx.MultiDiGraph
    data_codeflaws:  dict
    Returns
    ----------
    param_name: type
          Short description
    '''

    filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path,
                                   data_codeflaws['container'],
                                   data_codeflaws['c_source'])
    nline_removed = remove_lib(filename)
    graph = cfg.CFG("temp.c")
    code = []
    with open("temp.c", 'r') as f:
        code = [line for line in f]

    nx_cfg, nx_ast, nx_cfg_ast = build_nx_graph_cfg_ast(graph, code)
    nx_cfg_cov = nx_cfg.copy()

    tests_list = list(data_codeflaws['test_verdict'].keys())

    for i, test in enumerate(tests_list):
        covfile = "{}/{}/{}.gcov".format(
            ConfigClass.codeflaws_data_path, data_codeflaws['container'], test)
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

    return nx_cfg, nx_ast, nx_cfg_ast, nx_cfg_cov


def build_nx_cfg_ast_coverage_codeflaws(data_codeflaws: dict):
    ''' Build networkx controlflow ast coverage heterograph codeflaws
    Parameters
    ----------
    cfg_ast_g:  nx.MultiDiGraph
    data_codeflaws:  dict
    Returns
    ----------
    param_name: type
          Short description
    '''

    filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path,
                                   data_codeflaws['container'],
                                   data_codeflaws['c_source'])
    nline_removed = remove_lib(filename)
    graph = cfg.CFG("temp.c")

    with open("temp.c", 'r') as f:
        code = [line for line in f]
    nx_cfg, nx_ast, cfg_ast_g = build_nx_graph_cfg_ast(graph, code)
    nx_cfg_ast = cfg_ast_g.copy()

    tests_list = list(data_codeflaws['test_verdict'].keys())

    for i, test in enumerate(tests_list):
        covfile = "{}/{}/{}.gcov".format(
            ConfigClass.codeflaws_data_path, data_codeflaws['container'], test)
        coverage_map = get_coverage(covfile, nline_removed)
        test_node = cfg_ast_g.number_of_nodes()
        cfg_ast_g.add_node(test_node, name=f'test_{i}',
                           ntype='test', graph='test')
        for node in cfg_ast_g.nodes():
            # Check the line
            if cfg_ast_g.nodes[node]['graph'] != 'cfg':
                continue
            # Get corresponding lines
            start = cfg_ast_g.nodes[node]['start_line']
            end = cfg_ast_g.nodes[node]['end_line']
            if end - start > 0:     # This is a parent node
                continue

            for line in coverage_map:
                if line == start:
                    # The condition of parent node passing is less strict
                    if coverage_map[line] > 0:
                        cfg_ast_g.add_edge(
                            node, test_node, label='c_pass_test')
                        for ast_node in neighbors_out(
                                node, cfg_ast_g,
                                filter_func=lambda u, v, k, e: e['label'] ==
                                'corresponding_ast'):
                            cfg_ast_g.add_edge(
                                ast_node, test_node, label='a_pass_test')
                    else:
                        # 2 case, since a common parent line might only have
                        # 1 line
                        cfg_ast_g.add_edge(
                            node, test_node, label='c_fail_test')
                        for ast_node in neighbors_out(
                                node, cfg_ast_g,
                                filter_func=lambda u, v, k, e: e['label'] ==
                                'corresponding_ast'):
                            cfg_ast_g.add_edge(
                                ast_node, test_node, label='a_fail_test')

    return nx_cfg, nx_ast, nx_cfg_ast, cfg_ast_g
