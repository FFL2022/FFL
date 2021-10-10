from utils.utils import ConfigClass
from cfg import cfg
from utils.traverse_utils import build_nx_cfg, build_nx_ast_base,\
    build_nx_ast_full
from utils.preprocess_helpers import get_coverage, remove_lib
from graph_algos.nx_shortcuts import neighbors_in, neighbors_out
from codeflaws.data_format import key2bug, key2fix, key2bugfile,\
    key2fixfile, key2test_verdict, get_gcov_file
from graph_algos.nx_shortcuts import combine_multi, neighbors_out
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


def get_tree_diff(path1: str, path2: str):
    # Note: Please change to JAVA 11 before running this
    if path1.endswith('java'):
        cmd = ConfigClass.ast_java_command + path1 + ' ' + path2
    elif path1.endswith('c') or path1.endswith('cpp') or path1.endswith('cxx'):
        cmd = ConfigClass.ast_cpp_command + path1 + ' ' + path2
    json_str = '\n'.join(os.popen(cmd).read().split("\n")[1:])  # ignore 1st l
    map_dict = json.loads(json_str)
    map_dict['mapping'] = {int(k): int(v) for k, v in map_dict['mapping'].items()}
    return map_dict


def get_non_inserted_ancestor(rev_map_dict, dst_n, nx_ast_dst):
    parents = neighbors_in(dst_n, nx_ast_dst)
    while(len(parents) > 0):
        parent = parents[0]
        if parent not in rev_map_dict:
            parents = neighbors_in(parent, nx_ast_dst)
        else:
            return parent


def get_prev_sibs(u, q: nx.MultiDiGraph):
    parents = neighbors_in(u, q)
    if len(parents) > 0:
        p = parents[0]
        return neighbors_out(p, q, lambda _, x, k, e: x < u)
    return []


def get_next_sibs(u, q: nx.MultiDiGraph):
    parents = neighbors_in(u, q)
    if len(parents) > 0:
        p = parents[0]
        return neighbors_out(p, q, lambda _, x, k, e: x > u)
    return []


def add_placeholder_stmts_java(nx_ast):
    ''' add PlaceholderStatement for each block '''
    queue = [0]
    while (len(queue) > 0):
        node = queue.pop(0)
        if nx_ast.nodes[node]['ntype'] in ['Block', 'SwitchStatement',
                                           'IfStatement', 'ForStatement',
                                           'WhileStatement']:
            child_stmts = neighbors_out(node, nx_ast)
            new_node = max(nx_ast.nodes()) + 1
            if len(child_stmts) > 0:
                end_line = max([nx_ast.nodes[c]['end_line']
                                for c in child_stmts])
            else:
                end_line = nx_ast.nodes[node]['coord_line']
            nx_ast.add_node(new_node,
                            ntype='PlaceHolderStatement',
                            token='',
                            graph='ast',
                            coord_line=end_line,
                            end_line=end_line,
                            status=0
                            )
            nx_ast.add_edge(node, new_node, label='parent_child')
            child_stmts = child_stmts + [len(nx_ast.nodes())-1]
            queue.extend(child_stmts)
        else:
            queue.extend(neighbors_out(node, nx_ast))
    return nx_ast


def add_placeholder_stmts_cpp(nx_ast):
    ''' add PlaceholderStatement for each block '''
    queue = [0]
    while (len(queue) > 0):
        node = queue.pop(0)
        child_stmts = neighbors_out(
            node, nx_ast, lambda u, v, k, e: ConfigClass.check_is_stmt_cpp(
                          nx_ast.nodes[v]['ntype']))
        if len(child_stmts) > 0:
            child_stmts = neighbors_out(node, nx_ast)
            new_node = max(nx_ast.nodes()) + 1
            if len(child_stmts) > 0:
                end_line = max([nx_ast.nodes[c]['end_line']
                                for c in child_stmts])
            else:
                end_line = nx_ast.nodes[node]['coord_line']
            nx_ast.add_node(new_node,
                            ntype='placeholder_stmt',
                            token='',
                            graph='ast',
                            coord_line=end_line,
                            end_line=end_line,
                            status=0
                            )
            nx_ast.add_edge(node, new_node, label='parent_child')
            child_stmts = child_stmts + [len(nx_ast.nodes())-1]
            queue.extend(child_stmts)
        else:
            queue.extend(neighbors_out(node, nx_ast))
    return nx_ast


def check_statement_elem_removed(n, nx_ast, ldels,
                                 check_func=ConfigClass.check_is_stmt_java):
    queue = [n]
    started = False
    while len(queue) > 0:
        n = queue.pop(0)
        if check_func(nx_ast.nodes[n]['ntype']) and started:
            continue
        if n in ldels:
            return True
        queue.extend(neighbors_out(n, nx_ast))
        started = True
    return False


def check_statement_elem_inserted(n, nx_ast, lisrts,
                                  check_func=ConfigClass.check_is_stmt_java):
    queue = [n]
    started = False
    while len(queue) > 0:
        n = queue.pop(0)
        if check_func(nx_ast.nodes[n]['ntype']) and started:
            continue
        if n in lisrts:
            return True
        queue.extend(neighbors_out(n, nx_ast))
        started = True
    return False


def find_modified_statement(nx_ast_src, ldels,
                            check_func=ConfigClass.check_is_stmt_java):
    ns = [n for n in nx_ast_src.nodes()
          if check_func(nx_ast_src.nodes[n]['ntype'])]
    ns = [n for n in ns if check_statement_elem_removed(n, nx_ast_src, ldels,
                                                        check_func)]
    return ns


def find_inserted_statement(nx_ast_src, nx_ast_dst, rev_map_dict, lisrts,
                            check_func=ConfigClass.check_is_stmt_java):
    ns = [n for n in nx_ast_dst.nodes()
          if check_func(nx_ast_dst.nodes[n]['ntype'])]
    inserted_stmts = []
    # First, check if the statement itself is inserted
    ns_i = [n for n in ns if n in lisrts]
    for s_i in ns_i:
        if neighbors_in(s_i, nx_ast_dst)[0] in lisrts:
            continue
        dst_p = neighbors_in(s_i, nx_ast_dst)[0]
        dst_prev_sibs = get_prev_sibs(s_i, nx_ast_dst)
        dst_prev_sibs = [n for n in dst_prev_sibs if n not in lisrts]
        if len(dst_prev_sibs) > 0:
            src_prev_sib = rev_map_dict[max(dst_prev_sibs)]
            try:
                src_next_sib = min(get_next_sibs(src_prev_sib, nx_ast_src))
            except:
                print(nx_ast_dst.nodes[dst_p]['ntype'])
        else:
            # get the first child in the block
            src_next_sib = min(
                neighbors_out(rev_map_dict[dst_p], nx_ast_src))
        inserted_stmts.append(src_next_sib)

    ns_ni = [n for n in ns if n not in lisrts]
    for s_n in ns_ni:
        if check_statement_elem_inserted(s_n, nx_ast_dst, lisrts, check_func):
            inserted_stmts.append(rev_map_dict[s_n])
    return inserted_stmts


def build_nx_graph_stmt_annt(map_dict):
    ''' Statement-level annotation'''
    nx_ast_src = nx.MultiDiGraph()
    for ndict in sorted(map_dict['srcNodes'], key=lambda x: x['id']):
        nx_ast_src.add_node(ndict['id'], ntype=ndict['type'],
                            token=ndict['label'], graph='ast',
                            coord_line=ndict['range']['begin']['line'],
                            end_line=ndict['range']['end']['line'],
                            status=0)

        # Hypo 1: NX always keep the order of edges between one nodes and all
        # others, so we can recover sibling from this
        if ndict['parent_id'] != -1:
            nx_ast_src.add_edge(
                ndict['parent_id'], ndict['id'], label='parent_child')

    # Insert a place holder node for each

    nx_ast_dst = nx.MultiDiGraph()

    for ndict in sorted(map_dict['dstNodes'], key=lambda x: x['id']):
        nx_ast_dst.add_node(
            ndict['id'], ntype=ndict['type'], graph='ast',
            token=ndict['label'],
            coord_line=ndict['range']['begin']['line'],
            end_line=ndict['range']['begin']['line'],
            status=0
        )
        if ndict['parent_id'] != -1:
            nx_ast_dst.add_edge(
                ndict['parent_id'], ndict['id'], label='parent_child')
    nx_ast_src = add_placeholder_stmts_cpp(nx_ast_src)
    rev_map_dict = {v: k for k, v in map_dict['mapping'].items()}

    # moved nodes: check parent
    for nsid, ndid in map_dict['mapping'].items():
        psids = neighbors_in(nsid, nx_ast_src)
        pdids = neighbors_in(ndid, nx_ast_dst)
        if len(psids) > 0 and len(pdids) > 0:
            psid = psids[0]
            pdid = pdids[0]
            if psid in map_dict['mapping'] and pdid in rev_map_dict:
                if map_dict['mapping'][psid] != pdid:   # Moved
                    map_dict['deleted'].append(nsid)
                    map_dict['inserted'].append(ndid)

    for st_n in find_modified_statement(nx_ast_src, map_dict['deleted'],
                                        check_func=ConfigClass.check_is_stmt_cpp):
        nx_ast_src.nodes[st_n]['status'] = 1

    # inserted nodes: check sibling
    for st_n in find_inserted_statement(nx_ast_src, nx_ast_dst, rev_map_dict,
                                        map_dict['inserted'],
                                        check_func=ConfigClass.check_is_stmt_cpp):
        nx_ast_src.nodes[st_n]['status'] = 1

    return nx_ast_src, nx_ast_dst


def build_nx_graph_node_annt(map_dict, lang='java'):
    nx_ast_src = nx.MultiDiGraph()
    for ndict in sorted(map_dict['srcNodes'], key=lambda x: x['id']):
        nx_ast_src.add_node(ndict['id'], ntype=ndict['type'],
                            token=ndict['label'], graph='ast',
                            coord_line=ndict['range']['begin']['line'],
                            end_line=ndict['range']['end']['line'],
                            status=0)

        # Hypo 1: NX always keep the order of edges between one nodes and all
        # others, so we can recover sibling from this
        if ndict['parent_id'] != -1:
            nx_ast_src.add_edge(
                ndict['parent_id'], ndict['id'], label='parent_child')

    if lang == 'java':
        add_placeholder_func = add_placeholder_stmts_java
    elif lang == 'cpp':
        add_placeholder_func = add_placeholder_stmts_cpp

    nx_ast_src = add_placeholder_func(nx_ast_src)
    nx_ast_dst = nx.MultiDiGraph()

    for ndict in sorted(map_dict['dstNodes'], key=lambda x: x['id']):
        nx_ast_dst.add_node(
            ndict['id'], ntype=ndict['type'], graph='ast',
            token=ndict['label'],
            coord_line=ndict['range']['begin']['line'],
            end_line=ndict['range']['begin']['line'],
            status=0
        )
        if ndict['parent_id'] != -1:
            nx_ast_dst.add_edge(
                ndict['parent_id'], ndict['id'], label='parent_child')

    # get nx graph of two asts
    # deleted nodes
    for nid in map_dict['deleted']:
        nx_ast_src.nodes[nid]['status'] = 1

    # inserted nodes: check sibling
    rev_map_dict = {v: k for k, v in map_dict['mapping'].items()}
    for nid in map_dict['inserted']:
        nx_ast_dst.nodes[nid]['status'] = 2
        dst_p = get_non_inserted_ancestor(rev_map_dict, nid, nx_ast_dst)
        if dst_p is None:
            continue
        nx_ast_src.nodes[rev_map_dict[dst_p]]['status'] = 2
        # sibling
        dst_prev_sibs = get_prev_sibs(nid, nx_ast_dst)
        dst_prev_sibs = [n for n in dst_prev_sibs if n in rev_map_dict]
        if len(dst_prev_sibs) > 0:
            # Consider letting it to be 3
            nx_ast_src.nodes[rev_map_dict[max(dst_prev_sibs)]]['status'] = 1
            nx_ast_dst.nodes[max(dst_prev_sibs)]['status'] = 1
        else:
            src_p = rev_map_dict[dst_p]
            childs = neighbors_out(src_p, nx_ast_src)
            if len(childs) > 0:
                prev_sib = min(childs)
                nx_ast_src.nodes[prev_sib]['status'] = 1

    # moved nodes: check parent
    for nsid, ndid in map_dict['mapping'].items():
        psids = neighbors_in(nsid, nx_ast_src)
        pdids = neighbors_in(ndid, nx_ast_dst)
        if len(psids) > 0 and len(pdids) > 0:
            psid = psids[0]
            pdid = pdids[0]
            if psid in map_dict['mapping'] and pdid in rev_map_dict:
                if map_dict['mapping'][psid] != pdid:
                    # This node has been removed in the original tree
                    nx_ast_src.nodes[nsid]['status'] = 1
                    # and inserted in the new tree
                    nx_ast_dst.nodes[ndid]['status'] = 2

                    # reverse parent finding
                    to = rev_map_dict[pdid]
                    # Original destination node status is inserted
                    nx_ast_src.nodes[to]['status'] = 2
                    # Get the sibling before it
                    dst_prev_sibs = get_prev_sibs(ndid, nx_ast_dst)
                    dst_prev_sibs = [
                        n for n in dst_prev_sibs if n in rev_map_dict]
                    if len(dst_prev_sibs) > 0:
                        # consider letting it be 3
                        nx_ast_src.nodes[
                            rev_map_dict[max(dst_prev_sibs)]
                        ]['status'] = 1
                        nx_ast_dst.nodes[max(dst_prev_sibs)]['status'] = 1
                    else:
                        src_p = rev_map_dict[dst_p]
                        childs = neighbors_out(src_p, nx_ast_src)
                        if len(childs) > 0:
                            prev_sib = min(childs)
                            nx_ast_src.nodes[prev_sib]['status'] = 1

    return nx_ast_src, nx_ast_dst
