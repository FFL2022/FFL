import torch as th
import dgl
import networkx as nx
import sys
import os
from cfg import cfg, cfg2graphml, cfg_cdvfs_generator
from cfg.cfg_nodes import CFGNode
from pycparser import c_ast, plyparser
def get_coverage(filename, nline_removed):
    
    def process_line(line):
        tag, line_no, code = line.strip().split(':', 2)
        return tag.strip(), int(line_no.strip()), code
    
    coverage = {}
    with open(filename, "r") as f:
        gcov_file = f.read()
        for idx, line in enumerate(gcov_file.split('\n')):
            if idx <= 4 or len(line.strip()) == 0:
                continue
            
            try:
                tag, line_no, code = process_line(line)
            except:
                print('idx:', idx, 'line:', line)
                print(line.strip().split(':', 2))
                raise
            assert idx!=5 or line_no==1, gcov_file
        
            if tag == '-':
                continue
            elif tag == '#####':
                coverage[line_no - nline_removed] = 0
            else:  
                tag = int(tag) 
                coverage[line_no - nline_removed] = 1
        return coverage

def remove_lib(filename):
    count = 0
    with open(filename, "r") as f:
        with open("temp.c", "w") as t:
            for line in f:
                if line[0] != "#":
                    t.write(line)
                else:
                    count += 1
    return count

def traverse_cfg(node, parent, list_callfunction, list_callfuncline):
    tmp_n = {}
    tmp_e = {}
    start_line = node.get_start_line()
    last_line = node.get_last_line()
    if node._type == "END":
        return {}, {}
    if node._type == "CALL":
        x = node.get_ast_elem_list()
        for func in x:
            try:
                call_index = list_callfuncline[func.name.name]
                tmp_e[(last_line, call_index)] = 1
            except KeyError:
                pass
    tmp_e[(parent, start_line)] = 1
    for i in range(start_line, last_line + 1, 1):
        if i != last_line:
            tmp_e[(i, i+1)] = 1
        tmp_n[i] = node._type
    for child in node.get_children():
        n, e = traverse_cfg(child, last_line, list_callfunction, list_callfuncline)
        tmp_n.update(n)
        tmp_e.update(e)
    return tmp_n, tmp_e

def get_token(astnode, lower=True):
        if isinstance(astnode, str):
            return astnode.node
        name = astnode.__class__.__name__
        token = name
        is_name = False
        if is_leaf(astnode):
            attr_names = astnode.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = astnode.names[0]
                elif 'name' in attr_names:
                    token = astnode.name
                    is_name = True
                else:
                    token = astnode.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = astnode.declname
            if astnode.attr_names:
                attr_names = astnode.attr_names
                if 'op' in attr_names:
                    if astnode.op[0] == 'p':
                        token = astnode.op[1:]
                    else:
                        token = astnode.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

def is_leaf(astnode):
    if isinstance(astnode, str):
        return True
    return len(astnode.children()) == 0
        
def traverse_ast(node, index, parent_index):
    tmp_n = {}
    tmp_e = {}
    if parent_index != 0:
        tmp_e[(parent_index, index+1)] = 1
    index += 1
    curr_index = index
    tmp_n[index] = [get_token(node), node.coord.line]
    for edgetype, child in node.children():
        index, n, e = traverse_ast(child, index, curr_index)
        tmp_e.update(e)
        tmp_n.update(n)
    return index, tmp_n, tmp_e

def build_graph(problem_id, program_id, test_ids):
    filename = "/home/thanhlc/thanhlc/Data/nbl_dataset/sources/{}/{}.c".format(problem_id,program_id)
    
    print("======== CFG ========")
    list_cfg_nodes = {}
    list_cfg_edges = {}
    #Remove headers
    nline_removed = remove_lib(filename)
    
    # create CFG
    graph = cfg.CFG("temp.c")
    graph.make_cfg()
    graph.show()

    list_callfunction = [node._func_name for node in graph._entry_nodes]
    list_callfuncline = {}
    for i in range(len(graph._entry_nodes)):
       entry_node = graph._entry_nodes[i]
       list_cfg_nodes[entry_node.line] = "entry_node"
       list_callfuncline[entry_node._func_name] = entry_node.line
       if isinstance(entry_node._func_first_node, CFGNode):
            n, e = traverse_cfg(entry_node._func_first_node, entry_node.line, list_callfunction, list_callfuncline)
            list_cfg_nodes.update(n)
            list_cfg_edges.update(e)
    # print(list_cfg_nodes)
    # print(list_cfg_edges)
    print("Done !!!")
    print("======== AST ========")
    index = 0
    list_ast_nodes = {}
    list_ast_edges = {}
    ast = graph._ast
    for _, funcdef in ast.children():
        index, tmp_n, tmp_e = traverse_ast(funcdef, index, 0)
        list_ast_nodes.update(tmp_n)
        list_ast_edges.update(tmp_e)
        
    # print(list_ast_node)
    # print(list_ast_edge)
    print("Done !!!")
    print("======== Mapping AST-CFG ========")
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
    print("Done !!!")
    print("======== Mapping tests-CFG ========")
    for test in test_ids:
        covfile = "/home/thanhlc/thanhlc/Data/nbl_dataset/data/tests/{}/IN_{}.txt-{}.gcov".format(problem_id, test, program_id)
        cfg_to_tests[test] = get_coverage(covfile, nline_removed)
    
    # print(cfg_to_tests)
    
    print("======== Mapping tests-AST ========")
    ast_to_tests = {}
    
    for test in test_ids:
        ast_to_tests[test] = {}
        for line, ast_nodes in cfg_to_ast.items():
            for node in ast_nodes:
                try:
                    ast_to_tests[test][node] = cfg_to_tests[test][line]
                except KeyError:
                    pass
    # print(ast_to_tests)
    print("Done !!!")
    return list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests

def build_dgl_graph(list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests):
    ast_id2idx = {}
    ast_idx2id = {}
    index = 0
    for id in list_ast_nodes.keys():
        ast_id2idx[id] = index
        ast_idx2id[index] = id
        index += 1
    
    cfg_id2idx = {}
    cfg_idx2id = {}
    index = 0
    for id in list_cfg_nodes.keys():
        cfg_id2idx[id] = index
        cfg_idx2id[index] = id
        index += 1

    test_id2idx = {}
    test_idx2id = {}
    index = 0
    for test_id in cfg_to_tests.keys():
        test_id2idx[test_id] = index
        test_idx2id[index] = test_id



    print("======== Buiding DGL Graph =========")
    ast_ast_l = []
    ast_ast_r = []
    for l, r in list_ast_edges:
        ast_ast_l.append(ast_id2idx[l])
        ast_ast_r.append(ast_id2idx[r])
    # num_nodes = len(list_ast_nodes.keys())

    cfg_cfg_l = []
    cfg_cfg_r = []
    for l, r in list_cfg_edges:
        cfg_cfg_l.append(cfg_id2idx[l])
        cfg_cfg_r.append(cfg_id2idx[r])

    ast_cfg_l = []
    ast_cfg_r = []
    for cfg_node, ast_nodes in cfg_to_ast.items():
        for node in ast_nodes:
            ast_cfg_l.append(ast_id2idx[node])
            ast_cfg_r.append(cfg_id2idx[cfg_node])

    ast_test_l = []
    ast_test_r = []
    for id, ast_nodes in ast_to_tests.items():
        for node, link in ast_nodes.items():
            if link == 1:
                ast_test_l.append(ast_id2idx[node])
                ast_test_r.append(test_id2idx[id])

    cfg_test_l = []
    cfg_test_r = []
    for id, cfg_nodes in cfg_to_tests.items():
        for node, link in cfg_nodes.items():
            if link == 1:
                cfg_test_l.append(cfg_id2idx[node])
                cfg_test_r.append(test_id2idx[id])   

    graph_data = {
        ('cfg', 'cfglink_for', 'cfg'): (th.tensor(cfg_cfg_l), th.tensor(cfg_cfg_r)),
        ('cfg', 'cfglink_back', 'cfg'): (th.tensor(cfg_cfg_r), th.tensor(cfg_cfg_l)),
        ('cfg', 'ctlink', 'test'): (th.tensor(cfg_test_l), th.tensor(cfg_test_r)),
        ('test', 'tclink', 'cfg'): (th.tensor(cfg_test_r), th.tensor(cfg_test_l))
    }
    g = dgl.heterograph(graph_data)
    
    #CFG Featgit init
    cfg_feats = [None] * g.num_nodes("cfg")
    for key, feat in list_cfg_nodes.items():
        cfg_feats[cfg_id2idx[key]] = feat
    print(cfg_feats)
    print("Done !!!")

    return g, ast_id2idx, cfg_id2idx, test_id2idx

if __name__ == '__main__':
    list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests = build_graph("3055", "1049262", [ "40112","40439"])
    G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests)
