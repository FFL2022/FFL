from utils.utils import ConfigClass
from cfg import cfg, cfg2graphml, cfg_cdvfs_generator
from utils.traverse_utils import build_nx_cfg, build_nx_ast
import networkx as nx


def build_nx_graph(graph):
    list_cfg_nodes = {}
    list_cfg_edges = {}
    # create CFG
    graph = cfg.CFG("temp.c")
    graph.make_cfg()
    # graph.show()
    nx_cfg, cfg2nx = build_nx_cfg(graph)
    # print(list_cfg_nodes)
    # print(list_cfg_edges)
    # print("Done !!!")
    # print("======== AST ========")
    index = 0
    list_ast_nodes = {}
    list_ast_edges = {}
    nx_g = nx.MultiDiGraph()
    ast = graph.get_ast()
    nx_ast, ast2nx = build_nx_ast
    # Note: We are removing both include and global variables
    # Is there any way to change this

    # print(list_ast_nodes)
    # print(list_ast_edges)
    # print("Done !!!")
    # print("======== Mapping AST-CFG ========")
    cfg_to_ast = {}
    for id, value in list_ast_nodes.items():
        _, line = value
        try:
            cfg_to_ast[line].append(id)
        except KeyError:
            cfg_to_ast[line] = []
    # print(cfg_to_ast)
    with open("temp.c") as f:
        index = 1
        for line in f:
            index +=1

    os.remove("temp.c")
    cfg_to_tests = {}
    # print("Done !!!")
    # print("======== Mapping tests-CFG ========")
    if codeflaws == None:
        tests_list = list(nbl['test_verdict'].keys())

        for test in tests_list:
            covfile = "{}/{}/{}-{}.gcov".format(ConfigClass.nbl_test_path, nbl['problem_id'], test, nbl['program_id'])
            cfg_to_tests[test] = get_coverage(covfile, nline_removed)

        # print("======== Mapping tests-AST ========")
        ast_to_tests = {}

        for test in tests_list:
            ast_to_tests[test] = {}
            for line, ast_nodes in cfg_to_ast.items():
                for node in ast_nodes:
                    try:
                        ast_to_tests[test][node] = cfg_to_tests[test][line]
                    except KeyError:
                        pass

    else:
        tests_list = list(codeflaws['test_verdict'].keys())

        for test in tests_list:
            covfile = "{}/{}/{}.gcov".format(ConfigClass.codeflaws_data_path, codeflaws['container'], test)
            cfg_to_tests[test] = get_coverage(covfile, nline_removed)

        # print("======== Mapping tests-AST ========")
        ast_to_tests = {}

        for test in tests_list:
            ast_to_tests[test] = {}
            for line, ast_nodes in cfg_to_ast.items():
                for node in ast_nodes:
                    try:
                        ast_to_tests[test][node] = cfg_to_tests[test][line]
                    except KeyError:
                        pass

    return list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests

