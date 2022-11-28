from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl

from nbl.utils import all_keys, get_nx_ast_stmt_annt_gumtree

from utils.utils import ConfigClass
from utils.gumtree_utils import GumtreeASTUtils
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from graph_algos.nx_shortcuts import nodes_where
import os
import random
import pickle as pkl
import json
import torch
import tqdm

from json import JSONDecodeError

class NBLGumtreeNxStatementDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_nbl):
        super().__init__(
                all_entries=all_keys,
                process_func=get_nx_ast_stmt_annt_gumtree,
                save_dir=save_dir,
                name='gumtree_stmt',
                special_attrs=[('stmt_nodes', 
                    lambda nx_g: nodes_where(
            nx_g, lambda x: GumtreeASTUtils.check_is_stmt_cpp(nx_g.nodes[x]['ntype'], graph='ast')
            ))])


class NBLGumtreeNxNodeDataset(AstNxDataset):
    def __init__(self,
                 save_dir=ConfigClass.preprocess_dir_nbl):
        super().__init__(
                all_entries=all_keys,
                process_func=get_nx_ast_node_annt_gumtree,
                save_dir=save_dir,
                name='gumtree_node',
                special_attrs=[])
