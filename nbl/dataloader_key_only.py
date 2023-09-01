from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from nbl.data_format import test_verdict

import os
import pickle as pkl
import tqdm
import json

from utils.utils import ConfigClass
from utils.preprocess_helpers import get_coverage, remove_lib
from utils.nx_graph_builder import augment_with_reverse_edge
from graph_algos.nx_shortcuts import nodes_where, where_node, edges_where
from nbl.utils import all_keys, eval_set, mapping_eval, most_failed_val_set
from utils.pyc_parser.pyc_differ import get_graph_diff
from graph_algos.nx_shortcuts import combine_multi, neighbors_out,\
        nodes_where, where_node, edges_where

from utils.graph_visitor import DirectedVisitor
from utils.data_utils import AstNxDataset
from pycparser.plyparser import ParseError
import torch
import random


def get_cfg_ast_cov(key):
    nx_a, nx_a2, nx_c1, nx_c2, nx_ca1, nline_removed1 =\
        get_graph_diff(key['b_fp'], key['f_fp'])
    pid = key['problem_id']
    vid = key['buggy']
    tests_list = list(test_verdict[pid][vid].keys())
    nx_cat = nx_ca1.copy()
    for i, test in enumerate(tests_list):
        link_type = 'pass' if test_verdict[pid][vid][test] == 1 else 'fail'
        covfile = f"{ConfigClass.nbl_test_path}/{pid}/{test}-{vid}.gcov"
        coverage_map = get_coverage(covfile, nline_removed1)
        t_n = nx_cat.number_of_nodes()
        nx_cat.add_node(t_n, name=f'test_{i}', ntype='test', graph='test')
        for n in nodes_where(nx_cat, graph='cfg'):
            start = nx_cat.nodes[n]['start_line']
            end = nx_cat.nodes[n]['end_line']
            if end - start > 0:     # This is a parent node
                continue
            # Get corresponding covered line.
            for line in filter(lambda x: x == start and coverage_map[x] > 0,
                               coverage_map):
                nx_cat.add_edge(n, t_n, label=f'c_{link_type}_test')
                start_n_asts = neighbors_out(
                    n, nx_cat,
                    lambda u, v, k, e: e['label'] == 'corresponding_ast')
                for a_n in DirectedVisitor(
                        nx_cat, start_n_asts,
                        lambda u, v, k, e: nx_cat.nodes[v]['graph'] == 'ast' and\
                                u != v and v != t_n):
                    nx_cat.add_edge(
                        a_n, t_n, label=f'a_{link_type}_test')
    return nx_cat


class NBLNxDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_nbl):
        super().__init__(
                all_entries=all_keys,
                process_func=lambda k: get_cfg_ast_cov(k),
                save_dir=save_dir, name='nbl_key_only',
                special_attrs=[
                    'ast_lb_d': lambda nx_g: nodes_where(nx_g, graph='ast', status=1),
                    'ast_lb_i': lambda nx_g: nodes_where(nx_g, graph='ast', status=2),
                    'cfg_lb': lambda nx_g: nodes_where(nx_g, graph='cfg', status=1)], post_process_func=del_all_status)
        self.cfg_etypes = ['parent_child', 'next', 'ref', 'func_call']
