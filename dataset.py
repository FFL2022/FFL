import torch as th
import dgl
import sys
import os
from cfg import cfg, cfg2graphml, cfg_cdvfs_generator
from cfg.cfg_nodes import CFGNode
from pycparser import c_ast, plyparser
import fasttext
import torch.nn.functional
import pygraphviz as pgv
import sqlite3
import numpy
import time
from utils.preprocess_helpers import remove_lib, get_coverage, traverse_cfg
from utils.traverse_utils import traverse_cfg, traverse_ast
import numpy as np
import subprocess
from tqdm import tqdm
import pickle as pkl
from transcoder import code_tokenizer
from coconut.tokenizer import Tokenizer
from utils.utils import ConfigClass
from sklearn.preprocessing import MultiLabelBinarizer


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

def tokenize(input, tokenizer_opt):
    if (tokenizer_opt == 1):
        tokenized_ast_feats = list(map(int, subprocess.run(["/home/minhld/tokenizer/src/tokenizer"], stdout=subprocess.PIPE, text=True, input=input).stdout.strip().split("\t")))
        ###One-hot encode then convert to tensor
        # one_hot_ast_feats = th.zeros(len(tokenized_ast_feats), max(tokenized_ast_feats)+1)
        # one_hot_ast_feats[th.arange(len(tokenized_ast_feats)), th.tensor(tokenized_ast_feats)] = 1
    elif (tokenizer_opt == 2):
        # Install needed dependencies: conda install -c powerai sacrebleu
        tokenized_ast_feats = code_tokenizer.tokenize_cpp(input)
    else:
        vocab = ()
        tokenized_ast_feats = Tokenizer.tokenize(input, vocab, add_if_not_exist=False)

    return tokenized_ast_feats

def build_dgl_graph(nbl=None, codeflaws=None, graph_opt = 1, tokenizer_opt = 2, model = None):
    ### Graph option
    # CFG + Test
    # CFG + Test + AST

    ### Tokenizer option
    # 1. A Thanh gui (https://github.com/dspinellis/tokenizer/)
    # 2. TransCoder (https://github.com/facebookresearch/TransCoder/blob/master/preprocessing/src/code_tokenizer.py)
    # 3. CoCoNuT (https://github.com/lin-tan/CoCoNut-Artifact/blob/master/fairseq-context/fairseq/tokenizer.py)
    if nbl != None:
        print("======== Buiding DGL Graph of {}-{}-{} =========".format(nbl['problem_id'], nbl['uid'], nbl['program_id']))
        test_verdict = nbl['test_verdict']
    if codeflaws != None:
        print("======== Buiding DGL Graph of {} =========".format(codeflaws['container']))
        test_verdict = codeflaws['test_verdict']
    if model != None:
        embedding_model = model

    list_cfg_nodes, list_cfg_edges, list_ast_nodes, list_ast_edges, cfg_to_ast, cfg_to_tests, ast_to_tests = build_graph(nbl, codeflaws)
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
                if test_verdict[id]:
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
                if test_verdict[id]:
                    cfg_ptest_l.append(cfg_id2idx[node])
                    cfg_ptest_r.append(test_id2idx[id])
                else:
                    cfg_ftest_l.append(cfg_id2idx[node])
                    cfg_ftest_r.append(test_id2idx[id])

    if nbl != None:
        filename = "{}/{}/{}.c".format(ConfigClass.nbl_source_path, nbl['problem_id'],nbl['program_id'])
    if codeflaws != None:
        filename = "{}/{}/{}.c".format(ConfigClass.codeflaws_data_path, codeflaws['container'], codeflaws['c_source'])

    if graph_opt == 1:
        graph_data = {
        ('cfg', 'cfglink_for', 'cfg'): (th.tensor(cfg_cfg_l), th.tensor(cfg_cfg_r)),
        ('cfg', 'cfglink_back', 'cfg'): (th.tensor(cfg_cfg_r), th.tensor(cfg_cfg_l)),
        ('cfg', 'cfg_passT_link', 'passing_test'): (th.tensor(cfg_ptest_l, dtype=torch.int32), th.tensor(cfg_ptest_r, dtype=torch.int32)),
        ('passing_test', 'passT_cfg_link', 'cfg'): (th.tensor(cfg_ptest_r, dtype=torch.int32), th.tensor(cfg_ptest_l, dtype=torch.int32)),
        ('cfg', 'cfg_failT_link', 'failing_test'): (th.tensor(cfg_ftest_l, dtype=torch.int32), th.tensor(cfg_ftest_r, dtype=torch.int32)),
        ('failing_test', 'failT_cfg_link', 'cfg'): (th.tensor(cfg_ftest_r, dtype=torch.int32), th.tensor(cfg_ftest_l, dtype=torch.int32))
        }

        g = dgl.heterograph(graph_data)
        #CFG_feats
        cfg_labels = [None] * g.num_nodes("cfg")

        for key, feat in list_cfg_nodes.items():
            cfg_labels[cfg_id2idx[key]] = ConfigClass.cfg_label_corpus.index(feat)

        cfg_label_feats = th.nn.functional.one_hot(
            th.LongTensor(cfg_labels),
            len(ConfigClass.cfg_label_corpus)
        )

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
            ('cfg', 'cfg_failT_link', 'failing_test'): (th.tensor(cfg_ftest_l, dtype=torch.int32), th.tensor(cfg_ftest_r, dtype=torch.int32)),
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
        cfg_labels = [None] * g.num_nodes("cfg")

        for key, feat in list_cfg_nodes.items():
            cfg_labels[cfg_id2idx[key]] = ConfigClass.cfg_label_corpus.index(feat)

        cfg_label_feats = th.nn.functional.one_hot(
            th.LongTensor(cfg_labels),
            len(ConfigClass.cfg_label_corpus)
        )

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

        if nbl != None:
            vocab_file = open('/home/minhld/GNN4FL/nbl_vocab.txt', 'r')
        if codeflaws != None:
            vocab_file = open('/home/minhld/GNN4FL/codeflaws_vocab.txt', 'r')
        vocab_dict = dict(list(line.split()) for line in vocab_file)

        ###One-hot encode then convert to tensor
        tokens_ast_feats = [tokenize(input=feat, tokenizer_opt=tokenizer_opt) for feat in ast_feats]
        # print("\n👉 tokens_ast_feats (%d)\n\t" % len(tokens_ast_feats))
        # print(tokens_ast_feats)
        token_ids = [[vocab_dict[token] for token in tokens_ast_feat] for tokens_ast_feat in tokens_ast_feats]
        # print("\n👉 token_ids (%d)\n\t" % len(token_ids))
        # print(token_ids)

        # Install needed dependencies: conda install -c conda-forge scikit-learn
        ast_label_feats = th.tensor(MultiLabelBinarizer(classes=list(vocab_dict.values())).fit_transform(token_ids))

        g.nodes["ast"].data['label'] = ast_label_feats
        g.nodes["ast"].data['content'] = torch.FloatTensor([embedding_model.get_sentence_vector(feat) for feat in ast_feats])
        # print(g.nodes["ast"])
        print("Done !!!")

    else:
        print("Invalid graph option")

    return g, ast_id2idx, cfg_id2idx, test_id2idx


if __name__ == '__main__':
    embedding_model = fasttext.load_model('/home/thanhlc/thanhlc/Data/c_pretrained.bin')

    ### NBL_debug
    with open("/home/thanhlc/thanhlc/Data/nbl_dataset/test_verdict.pkl", "rb") as f:
        all_nbl_test_verdict = pkl.load(f)
    with open("/home/thanhlc/thanhlc/Data/nbl_dataset/training_data.pkl", "rb") as f:
        training_data = pkl.load(f)
    with open("/home/thanhlc/thanhlc/Data/nbl_dataset/bug_lines_info.pkl", "rb") as f:
        bug_lines_info = pkl.load(f)
    with open('/home/minhld/codeflaws/test_verdict.pkl', 'rb') as handle:
        all_codeflaws_test_verdict = pkl.load(handle)

    # nbl = {}
    # nbl['problem_id'] = "3029"
    # nbl['uid'] = "u50747"
    # nbl['program_id'] = "1044240"
    # nbl['test_verdict'] = all_nbl_test_verdict["3029"][1044240]
    # G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(nbl=nbl, model=embedding_model)

    ### Codeflaws_debug
    # container = "474-A-bug-14683024-14683054"
    # info = container.split('-')
    # codeflaws = {}
    # codeflaws['container'] = container
    # codeflaws['c_source'] = info[0] + '-' + info[1] + '-' + info[3]
    # codeflaws['test_verdict'] = all_codeflaws_test_verdict["{}-{}".format(info[0], info[1])][info[3]]
    # G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(codeflaws=codeflaws, model=embedding_model)

    count, keyError, parseError, gcovMissingError = 0, 0, 0, 0
    for container in os.listdir("/home/minhld/codeflaws/data/"):
        if os.path.isdir("{}/".format(ConfigClass.codeflaws_data_path) + container) == False: continue
        info = container.split('-')
        codeflaws = {}
        codeflaws['container'] = container
        codeflaws['c_source'] = info[0] + '-' + info[1] + '-' + info[3]
        codeflaws['test_verdict'] = all_codeflaws_test_verdict["{}-{}".format(info[0], info[1])][info[3]]
        try:
            G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(codeflaws=codeflaws, model=embedding_model)
            count+=1
        except KeyError:
            keyError+=1
        except plyparser.ParseError:
            parseError+=1
        except FileNotFoundError:
            gcovMissingError+=1

    print("OK: ", count, "\nkeyError: ", keyError, "\nparseError: ", parseError, "\ngcovMissingError", gcovMissingError)
