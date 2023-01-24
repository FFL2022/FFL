from torch.utils.data import Dataset
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    CodeflawsCFLStatementGraphMetadata
from utils.data_utils import NxDataloader
from utils.utils import ConfigClass
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from typing import List
import os
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from graph_algos.nx_shortcuts import nodes_where
import networkx as nx
import torch.utils.data


class CodeflawsCFLPyGStatementDataset(Dataset):

    def __init__(self,
                 dataloader: NxDataloader,
                 meta_data: CodeflawsCFLStatementGraphMetadata,
                 ast_enc=None,
                 save_dir=ConfigClass.preprocess_dir_codeflaws,
                 name='pyg_cfl_stmt'):
        self.dataloader = dataloader
        self.meta_data = meta_data if meta_data else\
            CodeflawsCFLStatementGraphMetadata(dataloader.get_dataset())
        self.save_dir = save_dir
        self.vocab_dict = dict(
            tuple(line.split())
            for line in open('preprocess/codeflaws_vocab.txt', 'r'))
        self.name = name
        self.graph_save_path = f"{save_dir}/{name}.pkl"
        self.info_path = f"{save_dir}/{name}_info.pkl"
        self.ast_enc = ast_enc
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

    def has_cache(self):
        return os.path.exists(self.graph_save_path) &\
            os.path.exists(self.info_path)

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, i):
        return self.gs[i], self.gs_stmt_nodes[i].long()

    @staticmethod
    def nx_to_pyg(meta_data, nx_g, ast_enc, stmt_nodes):
        nx_g = augment_with_reverse_edge_cat(nx_g, meta_data.t_e_asts, [])
        ori_ns = list(nx_g.nodes())[:]
        nx_g = nx.convert_node_labels_to_integers(nx_g)
        new_ns = list(nx_g.nodes())[:]
        map_ns = {n: i for n, i in zip(ori_ns, new_ns)}
        ess = [[[], []] for i in range(len(meta_data.t_all))]
        for u, v, e in nx_g.edges(data=True):
            e['label'] = meta_data.t_all.index(e['label'])
            ess[e['label']][0].append(u)
            ess[e['label']][1].append(v)
        ess = [add_self_loops(torch.tensor(es).long())[0] for es in ess]
        data = Data(ess=ess)
        n_asts = nodes_where(nx_g, graph='ast')
        l_a = torch.tensor([
            meta_data.ntype2id[nx_g.nodes[n]['ntype']] for n in n_asts
        ]).long()
        if ast_enc is not None:
            data.c_a = torch.tensor([
                ast_enc(nx_g.nodes[n]['token']) for n in n_asts
            ]).float()
        # for cfg, it will be text
        data.lbl = torch.tensor([nx_g.nodes[n]['status'] for n in n_asts])
        ts = torch.tensor([0] * (len(nx_g.nodes()) - len(n_asts)))
        data.xs = [l_a, ts]
        return data, \
            torch.tensor(list(map_ns[n] for n in stmt_nodes)).int()

    def convert_from_nx_to_pyg(self, nx_g, stmt_nodes):
        return self.nx_to_pyg(self.meta_data, nx_g, self.ast_enc, stmt_nodes)

    def process(self):
        self.gs, self.gs_stmt_nodes = [], []
        for nx_g, stmt_nodes in self.dataloader:
            g, g_stmt_nodes = self.convert_from_nx_to_pyg(nx_g, stmt_nodes)
            self.gs.append(g)
            self.gs_stmt_nodes.append(g_stmt_nodes)

    def save(self):
        torch.save(self.gs, self.graph_save_path)
        torch.save(self.gs_stmt_nodes, self.info_path)

    def load(self):
        self.gs = torch.load(self.graph_save_path)
        self.gs_stmt_nodes = torch.load(self.info_path)
