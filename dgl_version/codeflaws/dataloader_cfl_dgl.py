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
from dgl_version.data_utils import numerize_graph


class CodeflawsCFLDGLStatementDataset(DGLDataset):
    def __init__(self, dataloader: NxDataloader,
                 meta_data: CodeflawsAstGraphMetadata,
                 name: str,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = name
        self.graph_save_path = os.path.join(
            save_dir, f'dgl_{self.name}.bin')
        self.info_path = os.path.join(
            save_dir, f'dgl_{self.name}_info.pkl')
        self.dataloader = dataloader
        self.meta_data = meta_data
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsCFLDGLStatementDataset, self).__init__(
            name='codeflaws_dgl', url=None,
            raw_dir=".", save_dir=save_dir,
            force_reload=False, verbose=False)

        self.train()

    def has_cache(self):
        return os.path.exists(self.graph_save_path) and\
            os.path.exists(self.info_path)

    def load(self):
        self.gs = load_graphs(self.graph_save_path)[0]
        self.meta_graph = self.meta_data.meta_graph
        for k, v in pkl.load(open(self.info_path, 'rb')):
            setattr(self, k, v)
        self.train()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)

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

    def process(self):
        self.meta_graph = self.meta_data.meta_graph
        self.gs = []
        bar = tqdm.tqdm(self.dataloader)
        bar.set_description("Converting NX to DGL")
        self.stmt_idxs = []
        for i, (nx_g, stmt_nodes) in enumerate(bar):
            g, stmt_idx = self.convert_from_nx_to_dgl(nx_g, stmt_nodes)
            self.stmt_idxs.append(torch.tensor(stmt_idx).long())
            self.gs.append(g)

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, i):
        return self.gs[i], self.stmt_idxs[i]
