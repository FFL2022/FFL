from utils.utils import ConfigClass
from utils.pyc_utils import pyc_check_is_stmt_cpp, get_nx_ast_stmt_annt_pyc
from graph_algos.nx_shortcuts import nodes_where, edges_where,\
        where_node
import os
import pickle as pkl
import tqdm
from pycparser.plyparser import ParseError
from utils.data_utils import AstNxDataset, AstGraphMetadata
from utils.get_bug_localization import get_asts_mapping
from nbl.utils import all_keys, eval_set, mapping_eval, most_failed_val_set
from nbl.data_utils import get_nx_ast_stmt_annt_cfl_nbl


class NBLPyGCFLNxStatementDataset(AstNxDataset):

    def __init__(self, save_dir=ConfigClass.preprocess_dir_nbl):
        super().__init__(all_keys, get_nx_ast_stmt_annt_cfl_nbl,
                         save_dir, 'nbl_cfl_pyc_stmt',
                         [('stmt_nodes', lambda nx_g: nodes_where(
                             nx_g,
                             lambda x: pyc_check_is_stmt_cpp(nx_g.nodes[x]),
                             graph='ast'))])
