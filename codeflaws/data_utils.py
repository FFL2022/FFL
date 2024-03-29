import os
from utils.utils import ConfigClass
from codeflaws.data_format import key2bug, key2bugfile,\
    key2fixfile, key2test_verdict, get_gcov_file, test_verdict
from graph_algos.nx_shortcuts import neighbors_out, neighbors_in, \
        nodes_where, where_node, edges_where
from utils.pyc_parser.pyc_differ import get_graph_diff
from utils.preprocess_helpers import get_coverage, remove_lib
from utils.nx_graph_builder import build_nx_graph_cfg_ast
from utils.gumtree_utils import GumtreeBasedAnnotation, GumtreeASTUtils
from utils.pyc_utils import (pyc_check_is_stmt_cpp, get_nx_ast_stmt_annt_pyc,
                             check_statement_elem_inserted,
                             get_coverage_graph_ast_pyc)
import pickle as pkl

root = ConfigClass.codeflaws_data_path


def make_codeflaws_dict(key, test_verdict):
    info = key.split("-")
    codeflaws = {}
    codeflaws['container'] = key
    codeflaws['c_source'] = key2bug(key) + ".c"
    codeflaws['test_verdict'] = test_verdict["{}-{}".format(info[0],
                                                            info[1])][info[3]]
    return codeflaws


def get_all_keys():
    data = {}
    ''' Example line:
    71-A-bug-18359456-18359477	DCCR	WRONG_ANSWER	DIFFOUT~~Loop~If~Printf
    Diname=key                      ..?
    '''

    with open(f"{root}/codeflaws-defect-detail-info.txt", "rb") as f:
        for line in f:
            info = line.split()
            try:
                data[info[1].decode("utf-8")].append(info[0].decode("utf-8"))
            except:
                data[info[1].decode("utf-8")] = []

    # diname -> graph
    all_keys = []
    graph_data_map = {}
    for _, keys in data.items():
        for key in keys:
            if not os.path.isdir("{}/{}".format(root, key)):
                continue
            all_keys.append(key)
    return all_keys


def get_coverage_graph_cfg(key: str, nx_cfg, nline_removed):
    nx_cfg_cov = nx_cfg.copy()

    tests_list = key2test_verdict(key)

    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        coverage_map = get_coverage(covfile, nline_removed)
        test_node = nx_cfg_cov.number_of_nodes()
        nx_cfg_cov.add_node(test_node,
                            name=f'test_{i}',
                            ntype='test',
                            graph='test')
        link_type = 'pass' if tests_list[test] > 0 else 'fail'
        for node in nodes_where(nx_cfg_cov, graph='cfg'):
            # Get corresponding lines
            start = nx_cfg_cov.nodes[node]['start_line']
            end = nx_cfg_cov.nodes[node]['end_line']
            if end - start > 0:  # This is a parent node
                continue

            for line in coverage_map:
                if line == start and coverage_map[line] > 0:
                    nx_cfg_cov.add_edge(node,
                                        test_node,
                                        label=f'c_{link_type}_test')
    return nx_cfg_cov


def get_coverage_graph_ast(key, nx_ast_g, nline_removed):
    nx_ast_g = nx_ast_g.copy()

    tests_list = key2test_verdict(key)

    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        # fail/pass = neg in covfile
        coverage_map = get_coverage(covfile, nline_removed)
        test_node = max(nx_ast_g.nodes()) + 1
        nx_ast_g.add_node(test_node,
                          name=f'test_{i}',
                          ntype='test',
                          graph='test')
        link_type = 'fail' if 'neg' in covfile else 'pass'
        for line in coverage_map:
            if coverage_map[line] <= 0:
                continue
            for a_node in nx_ast_g.nodes:
                if nx_ast_g.nodes[a_node]['graph'] != 'ast':
                    continue
                if nx_ast_g.nodes[a_node]['start_line'] == line:
                    queue = [a_node]
                    while len(queue) > 0:
                        node = queue.pop()
                        if len(
                                neighbors_out(
                                    node, nx_ast_g,
                                    lambda u, v, k, e: v == test_node)) > 0:
                            # Visited
                            continue
                        nx_ast_g.add_edge(node,
                                          test_node,
                                          label=f'a_{link_type}_test')
                        queue.extend(
                            neighbors_out(
                                node, nx_ast_g, lambda u, v, k, e: nx_ast_g.
                                nodes[v]['graph'] == 'ast'))

    return nx_ast_g


def get_coverage_graph_cfg_ast(key: str, nx_cfg_ast, nline_removed):
    nx_cat = nx_cfg_ast.copy()  # CFG AST Test, CAT

    tests_list = key2test_verdict(key)

    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        coverage_map = get_coverage(covfile, nline_removed)
        t_n = nx_cat.number_of_nodes()
        nx_cat.add_node(t_n, name=f'test_{i}', ntype='test', graph='test')
        link_type = 'fail' if 'neg' in covfile else 'pass'
        for c_n in nodes_where(nx_cat, graph='cfg'):
            # Get corresponding lines
            start = nx_cat.nodes[c_n]['start_line']
            end = nx_cat.nodes[c_n]['end_line']
            if end - start > 0:  # This is a parent node
                continue

            for line in coverage_map:
                if line == start:
                    # The condition of parent node passing is less strict
                    if coverage_map[line] <= 0:
                        continue
                    nx_cat.add_edge(c_n, t_n, label=f'c_{link_type}_test')
                    queue = neighbors_out(
                        c_n, nx_cat,
                        lambda u, v, k, e: e['label'] == 'corresponding_ast')
                    while len(queue) > 0:
                        a_n = queue.pop()
                        if len(
                                neighbors_out(
                                    a_n, nx_cat,
                                    lambda u, v, k, e: v == t_n)) > 0:
                            # Visited
                            continue
                        nx_cat.add_edge(a_n, t_n, label=f'a_{link_type}_test')
                        queue.extend(
                            neighbors_out(
                                a_n, nx_cat, lambda u, v, k, e: nx_cat.nodes[v]
                                ['graph'] == 'ast'))
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


def get_cfg_ast_cov(key):
    nx_ast, nx_ast2, nx_cfg, nx_cfg2, nx_cfg_ast, nline_removed1 =\
        get_graph_diff(key2bugfile(key),
                       key2fixfile(key))
    nx_cfg_ast_cov = get_coverage_graph_cfg_ast(key, nx_cfg_ast,
                                                nline_removed1)
    return nx_ast, nx_ast2, nx_cfg, nx_cfg2, nx_cfg_ast, nx_cfg_ast_cov


if os.path.exists(ConfigClass.codeflaws_all_keys):
    all_codeflaws_keys = pkl.load(open(ConfigClass.codeflaws_all_keys, 'rb'))
else:
    all_codeflaws_keys = get_all_keys()
    pkl.dump(all_codeflaws_keys, open(ConfigClass.codeflaws_all_keys, 'wb'))


def get_nx_ast_node_annt_gumtree(key):
    src_b = key2bugfile(key)
    src_f = key2fixfile(key)
    test_list = key2test_verdict(key)
    cov_maps = []
    verdicts = []
    for i, test in enumerate(test_list):
        covfile = get_gcov_file(key, test)
        cov_maps.append(get_coverage(covfile, 0))
        verdicts.append(test_list[test] > 0)

    return GumtreeBasedAnnotation.build_nx_ast_cov_annt(
        src_b, src_f, cov_maps, verdicts,
        GumtreeBasedAnnotation.build_nx_graph_node_annt)


def get_nx_ast_stmt_annt_gumtree(key):
    src_b = key2bugfile(key)
    src_f = key2fixfile(key)
    test_list = key2test_verdict(key)
    cov_maps = []
    verdicts = []
    for i, test in enumerate(test_list):
        covfile = get_gcov_file(key, test)
        cov_maps.append(get_coverage(covfile, 0))
        verdicts.append(test_list[test] > 0)

    return GumtreeBasedAnnotation.build_nx_ast_cov_annt(
        src_b, src_f, cov_maps, verdicts,
        GumtreeBasedAnnotation.build_nx_graph_stmt_annt)


def get_nx_ast_node_gumtree(key):
    src_b = key2bugfile(key)
    test_list = key2test_verdict(key)
    cov_maps = []
    verdicts = []
    for i, test in enumerate(test_list):
        covfile = get_gcov_file(key, test)
        cov_maps.append(get_coverage(covfile, 0))
        verdicts.append(test_list[test] > 0)

    return GumtreeBasedAnnotation.build_nx_ast_cov(src_b, cov_maps, verdicts)


def get_nx_ast_stmt_gumtree(key):
    src_b = key2bugfile(key)
    test_list = key2test_verdict(key)
    cov_maps = []
    verdicts = []
    for i, test in enumerate(test_list):
        covfile = get_gcov_file(key, test)
        cov_maps.append(get_coverage(covfile, 0))
        verdicts.append(test_list[test] > 0)

    return GumtreeBasedAnnotation.build_nx_ast_cov(src_b, cov_maps, verdicts)


def get_non_inserted_ancestor(rev_map_dict, dst_n, nx_ast_dst):
    parents = neighbors_in(dst_n, nx_ast_dst)
    while (len(parents) > 0):
        parent = parents[0]
        if parent not in rev_map_dict:
            parents = neighbors_in(parent, nx_ast_dst)
        else:
            return parent


def find_inserted_statement_gumtree(nx_ast_src, nx_ast_dst, rev_map_dict,
                                    lisrts):
    ns = [
        n for n in nx_ast_dst.nodes()
        if pyc_check_is_stmt_cpp(nx_ast_dst.nodes[n])
    ]
    inserted_stmts = []
    # First, check if the statement itself is inserted
    ns_i = [n for n in ns if n in lisrts]
    for s_i in ns_i:
        if neighbors_in(s_i, nx_ast_dst)[0] in lisrts:
            continue
        dst_p = neighbors_in(s_i, nx_ast_dst)[0]
        dst_prev_sibs = GumtreeASTUtils.get_prev_sibs(s_i, nx_ast_dst)
        dst_prev_sibs = [n for n in dst_prev_sibs if n not in lisrts]
        # src_p = rev_map_dict[dst_p]
        # for n in neighbors_out(src_p, nx_ast_src):
        #    print(nx_ast_src.nodes[n])
        if len(dst_prev_sibs) > 0:
            src_prev_sib = rev_map_dict[max(dst_prev_sibs)]
            src_prev_sib = src_prev_sib if type(
                src_prev_sib) == int else src_prev_sib[0]
            try:
                print(src_prev_sib,
                      GumtreeASTUtils.get_next_sibs(src_prev_sib, nx_ast_src))
                src_next_sib = min(
                    GumtreeASTUtils.get_next_sibs(src_prev_sib, nx_ast_src))
            except:
                raise
                print(nx_ast_dst.nodes[dst_p]['ntype'])
        else:
            # get the first child in the block
            src_prev_sib = rev_map_dict[dst_p]
            src_prev_sib = src_prev_sib if type(
                src_prev_sib) == int else src_prev_sib[0]
            src_next_sib = min(neighbors_out(src_prev_sib, nx_ast_src))
        inserted_stmts.append(src_next_sib)

    ns_ni = [n for n in ns if n not in lisrts]
    for s_n in ns_ni:
        if check_statement_elem_inserted(s_n, nx_ast_dst, lisrts):
            inserted_stmts.append(rev_map_dict[s_n])
    return inserted_stmts


def get_nx_ast_stmt_annt_cfl(key):
    src_b = key2bugfile(key)
    src_f = key2fixfile(key)
    test_list = key2test_verdict(key)
    cov_maps = []
    verdicts = []
    for i, test in enumerate(test_list):
        covfile = get_gcov_file(key, test)
        cov_maps.append(get_coverage(covfile, 0))
        verdicts.append(test_list[test] > 0)
    return get_coverage_graph_ast_pyc(
            get_nx_ast_stmt_annt_pyc(src_b, src_f), cov_maps, verdicts)
