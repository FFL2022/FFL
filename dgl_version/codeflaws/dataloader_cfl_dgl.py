from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
import os
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    CodeflawsAstGraphMetadata
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from graph_algos.nx_shortcuts import nodes_where, edges_where,\
        where_node_not
from utils.data_utils import NxDataloader
import pickle as pkl
import random
import torch
import tqdm
from collections import defaultdict
from dgl_version.data_utils import numerize_graph, CodeDGLDataset


class CodeflawsCFLDGLStatementDataset(CodeDGLDataset):
    def __init__(self, dataloader: NxDataloader,
                 meta_data: CodeflawsAstGraphMetadata,
                 mode: str,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = f"{mode}_codeflaws_dgl_statement"
        super().__init__(dataloader, meta_data, self.name, save_dir)

    def convert_from_nx_to_dgl(self, nx_g, stmt_nodes):
        nx_g = augment_with_reverse_edge_cat(nx_g, self.meta_data.t_e_asts, [])
        g, n2id = numerize_graph(nx_g, ['ast', 'test'])
        ast2id = n2id['ast']
        # Create dgl ast node
        ast_labels = torch.tensor([
            self.meta_data.t_e_asts.index(nx_g.nodes[node]['ntype'])
            for node in n_asts], dtype=torch.long
        )

        g.nodes['ast'].data['label'] = ast_labels
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda x: ast2id[x], ast2id))] = torch.tensor([nx_g.nodes[x]['status'] for x in ast2id])
        g.nodes['ast'].data['tgt'] = ast_tgts
        stmt_idxs = [ast2id[n] for n in stmt_nodes]
        return g, stmt_idxs
