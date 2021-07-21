from utils.utils import ConfigClass
from cfg import cfg
from utils.traverse_utils import build_nx_cfg, build_nx_ast
from utils.preprocess_helpers import get_coverage, remove_lib
from graph_algos.nx_shortcuts import combine_multi, neighbors_out


def build_nx_graph_cfg_ast(graph):
    graph.make_cfg()
    ast = graph.get_ast()
    nx_cfg, cfg2nx = build_nx_cfg(graph)
    nx_ast, ast2nx = build_nx_ast(ast)

    for node in nx_cfg.nodes():
        nx_cfg.nodes[node]['graph'] = 'cfg'
    for node in nx_ast.nodes():
        nx_ast.nodes[node]['graph'] = 'ast'

    nx_h_g, batch = combine_multi([nx_ast, nx_cfg])
    for node in nx_h_g.nodes():
        if nx_h_g.nodes[node]['graph'] != 'cfg':
            continue
        # Get corresponding lines
        start = nx_h_g.nodes[node]['start_line']
        end = nx_h_g.nodes[node]['end_line']

        corresponding_ast_nodes = [n for n in nx_h_g.nodes()
                                   if nx_h_g.nodes[n]['graph'] == 'ast' and
                                   nx_h_g.nodes[n]['coord_line'] >= start and
                                   nx_h_g.nodes[n]['coord_line'] <= end]
        for ast_node in corresponding_ast_nodes:
            nx_h_g.add_edge(node, ast_node, label='corresponding_ast')
    return nx_cfg, nx_ast, nx_h_g


def build_nx_cfg_ast_coverage_codeflaws(data_codeflaws: dict):
    ''' Build networkx controlflow ast coverage heterograph codeflaws
    Parameters
    ----------
    cfg_ast_g:  nx.MultiDiGraph
          Short description
    data_codeflaws:  dict
          Short description
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
    nx_cfg, nx_ast, cfg_ast_g = build_nx_graph_cfg_ast(graph)
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

            for line in coverage_map:
                if line >= start and line <= end:
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
                    elif line == start and start == end:
                        cfg_ast_g.add_edge(
                            node, test_node, label='c_fail_test')
                        for ast_node in neighbors_out(
                                node, cfg_ast_g,
                                filter_func=lambda u, v, k, e: e['label'] ==
                                'corresponding_ast'):
                            cfg_ast_g.add_edge(
                                ast_node, test_node, label='a_fail_test')

    return nx_cfg, nx_ast, nx_cfg_ast, cfg_ast_g
