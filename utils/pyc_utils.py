from graph_algos.nx_shortcuts import neighbors_out, neighbors_in, \
        nodes_where, where_node, edges_where
from utils.pyc_parser.pyc_differ import get_asts_mapping
from utils.pyc_parser.cfg_ast_building_utils import convert_from_arity_to_rel


def pyc_check_is_stmt_cpp(node_dict: dict):
    return node_dict['ntype'] in [
            'Break', 'Return', 'Case', 'DoWhile', 'Decl', 'Default',
            'If', 'FuncCall', 'Continue', 'Goto', 'EmptyStatement',
            'While', 'ExprList', 'Switch', 'For', 'FuncDef'] and \
            node_dict['p_ntype'] in ['If', 'Else', 'For', 'Compound']


def get_nx_ast_stmt_annt_pyc(fp_b, fp_f):
    map_dict, nx_ast_src, nx_ast_dst = get_asts_mapping(fp_b, fp_f)
    nx_ast_src.nodes[0]['p_ntype'] = ''
    for n in nx_ast_src.nodes():
        p_ntype = nx_ast_src.nodes[n]['ntype']
        for cn in neighbors_out(n, nx_ast_src):
            nx_ast_src.nodes[cn]['p_ntype'] = p_ntype

        nx_ast_src.nodes[n]['status'] = 0
        if n in map_dict['deleted']:
            nx_ast_src.nodes[n]['status'] = 1
        if n in map_dict['inserted']:
            nx_ast_src.nodes[n]['status'] = 1

    nx_ast_dst.nodes[0]['p_ntype'] = ''
    for n in nx_ast_dst.nodes():
        p_ntype = nx_ast_dst.nodes[n]['ntype']
        for cn in neighbors_out(n, nx_ast_dst):
            nx_ast_dst.nodes[cn]['p_ntype'] = p_ntype
        nx_ast_dst.nodes[n]['status'] = 0

    nx_ast_src = pyc_add_placeholder_stmts_cpp(nx_ast_src)

    rev_map_dict = map_dict['rev_map_dict']

    for st_n in find_modified_statement(nx_ast_src, map_dict['deleted']):
        nx_ast_src.nodes[st_n]['status'] = 1

    # inserted nodes: check sibling
    # for st_n in find_inserted_statement(
    #         nx_ast_src, nx_ast_dst, rev_map_dict,
    #         map_dict['inserted']):
    #     # print(st_n)
    #     if type(st_n) == list:
    #         for _ in st_n:
    #             nx_ast_src.nodes[_]['status'] = 1
    #     elif type(st_n) == int:
    #         nx_ast_src.nodes[st_n]['status'] = 1
    #     else:
    #         raise

    nx_ast_src = convert_from_arity_to_rel(nx_ast_src)

    return nx_ast_src


def pyc_add_placeholder_stmts_cpp(nx_ast):
    ''' add PlaceholderStatement for each block '''
    queue = [0]
    while (len(queue) > 0):
        node = queue.pop(0)
        child_stmts = neighbors_out(
            node, nx_ast,
            lambda u, v, k, e: pyc_check_is_stmt_cpp(nx_ast.nodes[v]))
        if len(
                child_stmts
        ) > 0:  # or nx_ast.nodes[node]['ntype'] in ['block_content',  'case']:
            new_node = max(nx_ast.nodes()) + 1
            if len(child_stmts) > 0:
                end_line = max(
                    [nx_ast.nodes[c]['start_line'] for c in child_stmts])
            else:
                end_line = nx_ast.nodes[node]['start_line']
            n_orders = [nx_ast.nodes[n]['n_order'] for n in child_stmts]
            nx_ast.add_node(new_node,
                            ntype='placeholder_stmt',
                            p_ntype=nx_ast.nodes[node]['ntype'],
                            token='',
                            graph='ast',
                            start_line=end_line,
                            end_line=end_line,
                            n_order=0 if all(
                                item == 0
                                for item in n_orders) else max(n_orders) + 1,
                            status=0)
            labels = list(
                set(x[-1]['label'] for x in edges_where(
                    nx_ast, where_node(lambda x: x == node), where_node())))

            nx_ast.add_edge(
                node,
                new_node,
                label='parent_child' if len(labels) > 1 else labels[0])
            child_stmts = child_stmts + [len(nx_ast.nodes()) - 1]
        queue.extend(neighbors_out(node, nx_ast))
    return nx_ast


def find_modified_statement(nx_ast_src, ldels):
    ns = [
        n for n in nx_ast_src.nodes()
        if pyc_check_is_stmt_cpp(nx_ast_src.nodes[n])
    ]
    ns = [n for n in ns if check_statement_elem_removed(n, nx_ast_src, ldels)]
    return ns


def check_statement_elem_removed(n, nx_ast, ldels):
    queue = [n]
    started = False
    while len(queue) > 0:
        n = queue.pop(0)
        if pyc_check_is_stmt_cpp(nx_ast.nodes[n]) and started:
            continue
        if n in ldels:
            return True
        queue.extend(neighbors_out(n, nx_ast))
        started = True
    return False


def check_statement_elem_inserted(n, nx_ast, lisrts):
    queue = [n]
    started = False
    while len(queue) > 0:
        n = queue.pop(0)
        if pyc_check_is_stmt_cpp(nx_ast.nodes[n]) and started:
            continue
        if n in lisrts:
            return True
        queue.extend(neighbors_out(n, nx_ast))
        started = True
    return False


def get_coverage_graph_ast_pyc(nx_ast_g, cov_maps, verdicts):
    ''' Get coverage graph ast
        Parameters
        ----------
        cov_maps: each test case has coverage map: line -> freq,
        verdicts: each test case has 1 verdict: True/False
        Returns
        ----------
        nx_ast_g: nx_multi_di_graph
              Short description
    '''
    nx_ast_g = nx_ast_g.copy()
    for n, ndata in nx_ast_g.nodes(data=True):
        if 'graph' not in ndata:
            ndata['graph'] = 'ast'
    for i, (cov_map, verdict) in enumerate(zip(cov_maps, verdicts)):
        test_node = max(nx_ast_g.nodes()) + 1
        nx_ast_g.add_node(test_node, name=f'test_{i}',
                          ntype='test', graph='test')
        link_type = 'fail' if not verdict else 'pass'
        for line in cov_map:
            if cov_map[line] <= 0:
                continue
            for a_node in nodes_where(nx_ast_g, graph='ast'):
                if nx_ast_g.nodes[a_node]['start_line'] == line:
                    queue = [a_node]
                    while len(queue) > 0:
                        node = queue.pop()
                        if len(neighbors_out(node, nx_ast_g,
                                lambda u, v, k, e: v == test_node)
                        ) > 0:
                            # Visited
                            continue
                        nx_ast_g.add_edge(
                            node, test_node, label=f'a_{link_type}_t')
                        queue.extend(
                            neighbors_out(
                                node, nx_ast_g,
                                lambda u, v, k, e:
                                nx_ast_g.nodes[v]['graph'] == 'ast'))

    return nx_ast_g
