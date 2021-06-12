import torch as th
import dgl
import networkx as nx
import sys
import os
from cfg import cfg, cfg2graphml, cfg_cdvfs_generator
from cfg.cfg_nodes import CFGNode
from pycparser import c_ast, plyparser
import fasttext
import torch.nn.functional
# import pygraphviz as pgv
import sqlite3
import numpy
import time
# from utils.preprocess_helpers import make_dir_if_not_exists as mkdir, write_to_file as write, remove_lib, get_coverage
import numpy as np
import subprocess
from tqdm import tqdm
import pickle as pkl
from transcoder import code_tokenizer
from coconut.tokenizer import Tokenizer
import pickle as pickle
import json
from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils import ConfigClass

def get_coverage(filename, nline_removed):
    
    def process_line(line):
        tag, line_no, code = line.strip().split(':', 2)
        return tag.strip(), int(line_no.strip()), code
    
    coverage = {}
    with open(filename, "r") as f:
        gcov_file = f.read()
        for idx, line in enumerate(gcov_file.split('\n')):
            print(line)
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

def traverse_cfg(graph):
    list_cfg_nodes = {}
    list_cfg_edges = {}
    list_callfunction = [node._func_name for node in graph._entry_nodes]
    list_callfuncline = {}
    parent = {}
    is_traversed = []
    for i in range(len(graph._entry_nodes)):
       entry_node = graph._entry_nodes[i]
       list_cfg_nodes[entry_node.line] = "entry_node"
       list_callfuncline[entry_node._func_name] = entry_node.line
       is_traversed.append(entry_node)
       if isinstance(entry_node._func_first_node, CFGNode):
            queue = []
            node = entry_node._func_first_node
            queue.append(node)
            parent[node] = entry_node.line
            while len(queue) > 0:
                # print(queue)
                node = queue.pop(0)
                # print(node.get_start_line())
                if node not in is_traversed:
                    # print(node.get_start_line())
                    # print(node._type)
                    # print(node.get_children())
                    parent_id = parent[node]
                    is_traversed.append(node)
                    start_line = node.get_start_line()
                    last_line = node.get_last_line()
                    for child in node.get_children():
                        # print(child)
                        parent[child] = start_line
                        queue.append(child)
                    if node._type == "END":
                        pass
                    else:
                        if node._type == "CALL":
                            x = node.get_ast_elem_list()
                            for func in x:
                                try:
                                    call_index = list_callfuncline[func.name.name]
                                    list_cfg_edges[(last_line, call_index)] = 1
                                except KeyError:
                                    pass
                    
                        list_cfg_edges[(parent_id, start_line)] = 1
                        for i in range(start_line, last_line + 1, 1):
                            if i != last_line:
                                list_cfg_edges[(i, i+1)] = 1
                            list_cfg_nodes[i] = node._type
    return list_cfg_nodes, list_cfg_edges

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

def remove_lib(filename):
    count = 0
    with open(filename, "r") as f:
        with open("temp.c", "w") as t:
            for line in f:
                if (line.strip() == '') or (line.strip() != '' and line.strip()[0] != "#"):
                    t.write(line)
                else:
                    count += 1
    return count
        
def traverse_ast(node, index, parent, parent_index):
    tmp_n = {}
    tmp_e = {}
    if parent_index != 0:
        tmp_e[(parent_index, index+1)] = 1
    index += 1
    curr_index = index
    node_token = get_token(node)
    if node_token == "TypeDecl":
        coord_line = node.type.coord.line
    else:
        try: 
            coord_line = node.coord.line
        except AttributeError:
            coord_line = parent.coord.line

    tmp_n[index] = [node_token, coord_line]

    for edgetype, child in node.children():
        if child != None:
            index, n, e = traverse_ast(child, index, node, curr_index)
            tmp_e.update(e)
            tmp_n.update(n)
    return index, tmp_n, tmp_e

def build_graph(data, data_opt):
    # TODO: Duong dan den source code
    if data_opt == 'codeflaws':
        filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path, data['container'], data['c_source'])
    
    # print("======== CFG ========")
    list_cfg_nodes = {}
    list_cfg_edges = {}
    #Remove headers
    nline_removed = remove_lib(filename)
    print(nline_removed)

    # create CFG
    graph = cfg.CFG("temp.c")
    graph.make_cfg()
    # graph.show()
    list_cfg_nodes, list_cfg_edges = traverse_cfg(graph)
    # print(list_cfg_nodes)
    # print(list_cfg_edges)

    # print("======== AST ========")
    index = 0
    list_ast_nodes = {}
    list_ast_edges = {}
    ast = graph._ast
    for _, funcdef in ast.children():
        index, tmp_n, tmp_e = traverse_ast(funcdef, index, None, 0)
        list_ast_nodes.update(tmp_n)
        list_ast_edges.update(tmp_e)

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
    for test in list(data['test_verdict'].keys()):
        covfile = "{}/{}/{}.gcov".format(ConfigClass.codeflaws_data_path, data['container'], test)
        cfg_to_tests[test] = get_coverage(covfile, nline_removed)
    
    print(nline_removed, list_cfg_nodes)
    print(cfg_to_tests.values())
    
    # print("======== Mapping tests-AST ========")
    ast_to_tests = {}
    
    for test in list(data['test_verdict'].keys()):
        ast_to_tests[test] = {}
        for line, ast_nodes in cfg_to_ast.items():
            for node in ast_nodes:
                try:
                    ast_to_tests[test][node] = cfg_to_tests[test][line]
                except KeyError:
                    pass
    # print(ast_to_tests)
    # print("Done !!!")
    return list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests

def read_cfile(filename):
    pass


def build_dgl_graph(data, data_opt='nbl', graph_opt=2, tokenizer_opt=2, model=None):
    ### Graph option
    # CFG + Test 
    # CFG + Test + AST

    ### Tokenizer option
    # 1. A Thanh gui (https://github.com/dspinellis/tokenizer/)
    # 2. TransCoder (https://github.com/facebookresearch/TransCoder/blob/master/preprocessing/src/code_tokenizer.py)
    # 3. CoCoNuT (https://github.com/lin-tan/CoCoNut-Artifact/blob/master/fairseq-context/fairseq/tokenizer.py)
    # print("======== Buiding DGL Graph of {}.c =========".format(c_source))
    if model != None:
        embedding_model = model

    list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests = build_graph(data, data_opt)
    # print(list_cfg_nodes)
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

    print(cfg_id2idx, cfg_idx2id)

    test_id2idx = {}
    test_idx2id = {}
    index = 0
    for test_id in cfg_to_tests.keys():
        test_id2idx[test_id] = index
        test_idx2id[index] = test_id

    ast_ast_l = []
    ast_ast_r = []
    for l, r in list_ast_edges:
        ast_ast_l.append(ast_id2idx[l])
        ast_ast_r.append(ast_id2idx[r])

    cfg_cfg_l = []
    cfg_cfg_r = []
    for l, r in list_cfg_edges:
        cfg_cfg_l.append(cfg_id2idx[l])
        cfg_cfg_r.append(cfg_id2idx[r])

    ast_cfg_l = []
    ast_cfg_r = []
    
    for cfg_node, ast_nodes in cfg_to_ast.items():
        if cfg_node in cfg_id2idx:
            for node in ast_nodes:
                ast_cfg_l.append(ast_id2idx[node])
                ast_cfg_r.append(cfg_id2idx[cfg_node])

    ast_ftest_l = []
    ast_ftest_r = []
    ast_ptest_l = []
    ast_ptest_r = []
    for id, ast_nodes in ast_to_tests.items():
        for node, link in ast_nodes.items():
            if link == 1 and node in ast_id2idx:
                if data['test_verdict'][id]:
                    ast_ptest_l.append(ast_id2idx[node])
                    ast_ptest_r.append(test_id2idx[id])
                else:
                    ast_ftest_l.append(ast_id2idx[node])
                    ast_ftest_r.append(test_id2idx[id])
    cfg_ftest_l = []
    cfg_ftest_r = []
    cfg_ptest_l = []
    cfg_ptest_r = []
    for id, cfg_nodes in cfg_to_tests.items():
        for node, link in cfg_nodes.items():
            if link == 1 and node in cfg_id2idx:
                if data['test_verdict'][id]:
                    cfg_ptest_l.append(cfg_id2idx[node])
                    cfg_ptest_r.append(test_id2idx[id])
                else:
                    cfg_ftest_l.append(cfg_id2idx[node])
                    cfg_ftest_r.append(test_id2idx[id])  

    if graph_opt == 1:
        graph_data = {
        ('cfg', 'cfglink_for', 'cfg'): (th.tensor(cfg_cfg_l), th.tensor(cfg_cfg_r)),
        ('cfg', 'cfglink_back', 'cfg'): (th.tensor(cfg_cfg_r), th.tensor(cfg_cfg_l)),
        ('cfg', 'cfg_passT_link', 'passing_test'): (th.tensor(cfg_ptest_l, dtype=torch.int32), th.tensor(cfg_ptest_r, dtype=torch.int32)),
        ('passing_test', 'passT_cfg_link', 'cfg'): (th.tensor(cfg_ptest_r, dtype=torch.int32), th.tensor(cfg_ptest_l, dtype=torch.int32)),
        ('cfg', 'ctlink', 'cfg_failT_link'): (th.tensor(cfg_ftest_l, dtype=torch.int32), th.tensor(cfg_ftest_r, dtype=torch.int32)),
        ('failing_test', 'failT_cfg_link', 'cfg'): (th.tensor(cfg_ftest_r, dtype=torch.int32), th.tensor(cfg_ftest_l, dtype=torch.int32))
        }

        g = dgl.heterograph(graph_data)
        #CFG_feats
        cfg_label_corpus = ["entry_node", "COMMON", "IF", "ELSE", "ELSE_IF", "END_IF", "FOR", "WHILE", "DO_WHILE", "PSEUDO", "CALL", "END"]
        cfg_labels = [None] * g.num_nodes("cfg")
        for key, feat in list_cfg_nodes.items():
            cfg_labels[cfg_id2idx[key]] = cfg_label_corpus.index(feat)
        cfg_label_feats = th.nn.functional.one_hot(th.LongTensor(cfg_labels), len(cfg_label_corpus))

        if data_opt == 'codeflaws':
            filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path, data['container'], data['c_source'])
        code = []
        with open(filename, "r") as f:
            for line in f:
                if line[0] != "#":
                    code.append(line)
        cfg_content_feats = [None] * g.num_nodes("cfg")
        for key, feat in list_cfg_nodes.items():
            cfg_content_feats[cfg_id2idx[key]] = embedding_model.get_sentence_vector(code[key-1].replace("\n", ""))
    
        # (cfg_content_feats)
        g.nodes["cfg"].data['label'] = cfg_label_feats
        g.nodes["cfg"].data['content'] = torch.FloatTensor(cfg_content_feats)
        print("Done !!!")

    elif graph_opt == 2:
        graph_data = {
            ('cfg', 'cfglink_for', 'cfg'): (th.tensor(cfg_cfg_l), th.tensor(cfg_cfg_r)),
            ('cfg', 'cfglink_back', 'cfg'): (th.tensor(cfg_cfg_r), th.tensor(cfg_cfg_l)),
            ('cfg', 'cfg_passT_link', 'passing_test'): (th.tensor(cfg_ptest_l, dtype=torch.int32), th.tensor(cfg_ptest_r, dtype=torch.int32)),
            ('passing_test', 'passT_cfg_link', 'cfg'): (th.tensor(cfg_ptest_r, dtype=torch.int32), th.tensor(cfg_ptest_l, dtype=torch.int32)),
            ('cfg', 'ctlink', 'cfg_failT_link'): (th.tensor(cfg_ftest_l, dtype=torch.int32), th.tensor(cfg_ftest_r, dtype=torch.int32)),
            ('failing_test', 'failT_cfg_link', 'cfg'): (th.tensor(cfg_ftest_r, dtype=torch.int32), th.tensor(cfg_ftest_l, dtype=torch.int32)),

            ('ast', 'astlink_for', 'ast'): (th.tensor(ast_ast_l), th.tensor(ast_ast_r)),
            ('ast', 'astlink_back', 'ast'): (th.tensor(ast_ast_r), th.tensor(ast_ast_l)),
            ('ast', 'ast_passT_link', 'passing_test'): (th.tensor(ast_ptest_l, dtype=torch.int32), th.tensor(ast_ptest_r, dtype=torch.int32)),
            ('passing_test', 'passT_ast_link', 'ast'): (th.tensor(ast_ptest_r, dtype=torch.int32), th.tensor(ast_ptest_l, dtype=torch.int32)),
            ('ast', 'ast_failT_link', 'failing_test'): (th.tensor(ast_ftest_l, dtype=torch.int32), th.tensor(ast_ftest_r, dtype=torch.int32)),
            ('failing_test', 'failT_ast_link', 'ast'): (th.tensor(ast_ftest_r, dtype=torch.int32), th.tensor(ast_ftest_l, dtype=torch.int32)),

            ('ast', 'ast_cfg_link', 'cfg'): (th.tensor(ast_cfg_l), th.tensor(ast_cfg_r)),
            ('cfg', 'cfg_ast_link', 'ast'): (th.tensor(ast_cfg_r), th.tensor(ast_cfg_l))
        }

        g = dgl.heterograph(graph_data)
        #CFG_feats
        cfg_label_corpus = ["entry_node", "COMMON", "IF", "ELSE", "ELSE_IF", "END_IF", "FOR", "WHILE", "DO_WHILE", "PSEUDO", "CALL", "END"]
        cfg_labels = [None] * g.num_nodes("cfg")
        for key, feat in list_cfg_nodes.items():
            cfg_labels[cfg_id2idx[key]] = cfg_label_corpus.index(feat)
        cfg_label_feats = th.nn.functional.one_hot(th.LongTensor(cfg_labels), len(cfg_label_corpus))

        filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path, data['container'], data['c_source'])
        code = []
        with open(filename, "r") as f:
            for line in f:
                if line[0] != "#":
                    code.append(line)
        cfg_content_feats = [None] * g.num_nodes("cfg")
        for key, feat in list_cfg_nodes.items():
            cfg_content_feats[cfg_id2idx[key]] = embedding_model.get_sentence_vector(code[key-1].replace("\n", ""))
    
        # (cfg_content_feats)
        g.nodes["cfg"].data['label'] = cfg_label_feats
        g.nodes["cfg"].data['content'] = torch.FloatTensor(cfg_content_feats)
        # print(g.nodes["cfg"])

        #AST Feat
        ast_feats = [None] * g.num_nodes("ast")
        for key, value in list_ast_nodes.items():
            feat, _ = value
            ast_feats[ast_id2idx[key]] = feat

        ###Tokenize
        tokenized_ast_feats = tokenize(input=' '.join(ast_feats), option=2)
        tokens = list(dict.fromkeys(tokenized_ast_feats))

        ast_feats_tensor = one_hot_encode(ast_feats)
        # print("\n👉 input: ast_feats (%d)\n\t" % len(ast_feats))
        # print(ast_feats)
        # print("\n👉 one_hot_encode(ast_feats):", ast_feats_tensor.size(), ast_feats_tensor.dtype)
        # print("\n", ast_feats_tensor)

    else:
        print("Invalid graph option")

    return g, ast_id2idx, cfg_id2idx, test_id2idx, tokens

def tokenize(input, option):
    # 1. A Thanh gui (https://github.com/dspinellis/tokenizer/)
    # 2. TransCoder (https://github.com/facebookresearch/TransCoder/blob/master/preprocessing/src/code_tokenizer.py)
    # 3. CoCoNuT (https://github.com/lin-tan/CoCoNut-Artifact/blob/master/fairseq-context/fairseq/tokenizer.py)
    if (option == 1):
        tokenized_ast_feats = list(map(int, subprocess.run(["/home/minhld/tokenizer/src/tokenizer"], stdout=subprocess.PIPE, text=True, input=input).stdout.strip().split("\t")))
    elif (option == 2):
        # Dependencies: conda install -c powerai sacrebleu
        tokenized_ast_feats = code_tokenizer.tokenize_cpp(input)
    else:
        vocab = {}
        tokenized_ast_feats = Tokenizer.tokenize(input, vocab, add_if_not_exist=False)

    return tokenized_ast_feats

def get_file_names_with_strings(str, full_list):
    final_list = [nm for nm in full_list if str in nm]

    return final_list

def build_vocab_dict():
    embedding_model = fasttext.load_model('/home/thanhlc/thanhlc/Data/c_pretrained.bin')
    vocab_list = []
    with open('/home/minhld/codeflaws/test_verdict.pkl', 'rb') as handle:
        all_test_verdict = pickle.load(handle)
    # print(json.dumps(all_test_verdict, indent = 4))

    # conda install -c conda-forge tqdm
    count, keyError, parseError, gcovMissingError = 0, 0, 0, 0
    for dir in tqdm(os.listdir("{}/".format(ConfigClass.codeflaws_data_path)), desc="Tokenizing..."):
    # for dir in os.listdir("/home/minhld/codeflaws/data/"):
        if os.path.isdir("{}/".format(ConfigClass.codeflaws_data_path) + dir) == False: continue 
        info = dir.split('-')
        c_source = info[0] + '-' + info[1] + '-' + info[3]
        try:
            test_verdict = all_test_verdict["{}-{}".format(info[0], info[1])][info[3]]
            G, ast_id2idx, cfg_id2idx, test_id2idx, tokens = build_dgl_graph(dir, c_source, test_verdict, model=embedding_model)
            vocab_list = vocab_list + list(set(tokens) - set(vocab_list))
            count+=1
        except KeyError:
            keyError+=1
        except plyparser.ParseError:
            parseError+=1
        except FileNotFoundError:
            gcovMissingError+=1

    print("OK: ", count, "\nkeyError: ", keyError, "\nparseError: ", parseError, "\ngcovMissingError", gcovMissingError)
    
    if vocab_list:
        with open('/home/minhld/GNN4FL/codeflaws_vocab.txt', 'w') as file_handler:
            for index, item in enumerate(vocab_list):
                file_handler.write("{} {}\n".format(item, index + 1))

    return {k: v for v, k in enumerate(vocab_list)}  

def one_hot_encode(ast_feats, tokenizer_opt=2):
    # vocab_dict = build_vocab_dict()
    assert isinstance(ast_feats, list), "\n  Input is not a list\n"
    vocab_dict = {}
    vocab_file = open('/home/minhld/GNN4FL/codeflaws_vocab.txt', 'r')
    for line in vocab_file:
        key, value = line.split()
        vocab_dict[key] = value

    # print("\n====== vocab_dict (%d) ======" % len(vocab_dict))
    # print(vocab_dict)

    ###One-hot encode then convert to tensor
    tokens_ast_feats = [tokenize(input=feat, option=tokenizer_opt) for feat in ast_feats]
    token_ids = [[vocab_dict[token] for token in tokens_ast_feat] for tokens_ast_feat in tokens_ast_feats]
    # print("\n👉 tokens_ast_feats (%d)\n\t" % len(tokens_ast_feats))
    # print(tokens_ast_feats)
    # print("\n👉 token_ids (%d)\n\t" % len(token_ids))
    # print(token_ids)

    # conda install -c conda-forge scikit-learn
    return th.tensor(MultiLabelBinarizer(classes=list(vocab_dict.values())).fit_transform(token_ids))
    
if __name__ == '__main__':
    # vocab_dict = build_vocab_dict()

    embedding_model = fasttext.load_model('/home/thanhlc/thanhlc/Data/c_pretrained.bin')
    with open('/home/minhld/codeflaws/test_verdict.pkl', 'rb') as handle:
        all_test_verdict = pickle.load(handle)
    # print(json.dumps(all_test_verdict, indent = 4))

    container = "471-A-bug-17550066-17550110"
    info = container.split('-')
    data = {}
    data['container'] = container
    data['c_source'] = info[0] + '-' + info[1] + '-' + info[3]
    data['test_verdict'] = all_test_verdict["{}-{}".format(info[0], info[1])][info[3]]
    G, ast_id2idx, cfg_id2idx, test_id2idx, tokens = build_dgl_graph(
        data=data,
        data_opt='codeflaws',
        model=embedding_model
    )
