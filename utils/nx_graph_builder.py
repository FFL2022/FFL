from utils.utils import ConfigClass
from utils.preprocess_helpers import remove_lib

def build_graph(nbl=None, codeflaws=None):
    if nbl != None:
        filename = "{}/{}/{}.c".format(ConfigClass.nbl_source_path, nbl['problem_id'],nbl['program_id'])
    if codeflaws != None:
        filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path, codeflaws['container'], codeflaws['c_source'])

    # print("======== CFG ========")

    list_cfg_nodes = {}
    list_cfg_edges = {}
    #Remove headers
    nline_removed = remove_lib(filename)

    # create CFG
    graph = cfg.CFG("temp.c")
    graph.make_cfg()
    # graph.show()
    list_cfg_nodes, list_cfg_edges = traverse_cfg(graph)
    # print(list_cfg_nodes)
    # print(list_cfg_edges)
    # print("Done !!!")
    # print("======== AST ========")
    index = 0
    list_ast_nodes = {}
    list_ast_edges = {}
    ast = graph._ast
    for _, funcdef in ast.children():
        index, tmp_n, tmp_e = traverse_ast(funcdef, index, None, 0)
        list_ast_nodes.update(tmp_n)
        list_ast_edges.update(tmp_e)

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

