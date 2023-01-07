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
from utils.data_utils import AstNxDataset, AstGraphMetadata


class CodeflawsCFLNxStatementDataset(AstNxDataset):

    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(all_codeflaws_keys, get_nx_ast_stmt_annt_cfl,
                         save_dir, 'cfl_stmt',
                         [('stmt_nodes', lambda nx_g: nodes_where(
                             nx_g,
                             lambda x: cfl_check_is_stmt_cpp(nx_g.nodes[x]),
                             graph='ast'))])


CodeflawsAstGraphMetadata = AstGraphMetadata
CodeflawsCFLStatementGraphMetadata = AstGraphMetadata
