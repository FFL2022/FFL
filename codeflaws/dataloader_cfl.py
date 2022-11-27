from utils.utils import ConfigClass
from codeflaws.data_utils import all_codeflaws_keys,\
    get_nx_ast_stmt_annt_cfl, \
    cfl_check_is_stmt_cpp
from graph_algos.nx_shortcuts import nodes_where, edges_where,\
        where_node
import os
import pickle as pkl
import tqdm
from pycparser.plyparser import ParseError
from utils.data_utils import AstNxDataset


class CodeflawsCFLNxStatementDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(all_codeflaws_keys, get_nx_ast_stmt_annt_cfl,
                save_dir, 'cfl_stmt', [('stmt_nodes', nodes_where(
                    nx_g,
                    lambda x: cfl_check_is_stmt_cpp(nx_g.nodes[x]),
                    graph='ast')]

       
class CodeflawsCFLStatementGraphMetadata(object):
    def __init__(self, nx_g_dataset):
        self.t_asts = nx_g_dataset.ast_types
        self.ntype2id = {n: i for i, n in enumerate(self.t_asts)}
        self.t_e_asts = nx_g_dataset.ast_etypes
        self.meta_graph = self.construct_edge_metagraph()

    def construct_edge_metagraph(self):
        self.t_e_a_a = self.t_e_asts + \
            list(map(lambda x: f'{x}_reverse', self.t_e_asts))
        self.t_e_a_t = ['a_pass_t', 'a_fail_t']
        self.t_e_t_a = ['t_pass_a', 't_fail_a']
        self.t_all = (self.t_e_a_a + self.t_e_a_t + self.t_e_t_a)
        self.srcs = ['ast'] * len(self.t_e_a_a) +\
            ['ast'] * len(self.t_e_a_t) + ['test'] * len(self.t_e_t_a)
        self.dsts = ['ast'] * len(self.t_e_a_a) +\
            ['test'] * len(self.t_e_a_t) + ['ast'] * len(self.t_e_t_a)
        return list(zip(self.srcs, self.t_all, self.dsts))
