import networkx as nx
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from torch_geometric.utils import add_self_loops
import torch
from graph_algos.nx_shortcuts import nodes_where
from torch_geometric.data import Data
from utils.data_utils import NxDataloader, AstGraphMetadata
from utils.utils import ConfigClass
import os
import torch.utils.data
from torch.utils.data import Dataset


def get_node2id(nx_g, graphs=['ast', 'cfg', 'test']):
    n2id = {}
    for graph in graphs:
        ns = nodes_where(nx_g, graph=graph)
        n2id[graph] = dict([n, i] for i, n in enumerate(ns))
    return n2id


### TESTED TO WORK CORRECTLY, DO NOT TOUCH
def nx_to_pyg_stmt(meta_data, nx_g, ast_enc, stmt_nodes):
    nx_g = augment_with_reverse_edge_cat(nx_g, meta_data.t_e_asts, [])
    n2id = get_node2id(nx_g)
    ess = [[[], []] for i in range(len(meta_data.t_all))]
    for u, v, e in nx_g.edges(data=True):
        ugraph, vgraph = nx_g.nodes[u]['graph'], nx_g.nodes[v]['graph']
        et_idx = meta_data.meta_graph.index(
            (ugraph, e['label'].replace('test', 't'), vgraph))
        uid = n2id[ugraph][u]
        vid = n2id[vgraph][v]
        assert meta_data.meta_graph[et_idx][0] == ugraph, f"expected {meta_data.meta_graph[et_idx]}, instead got {ugraph}"
        assert meta_data.meta_graph[et_idx][2] == vgraph, f"expected {meta_data.meta_graph[et_idx]}, instead got {vgraph}"
        assert uid < len(n2id[ugraph])
        assert vid < len(n2id[vgraph])

        ess[et_idx][0].append(uid)
        ess[et_idx][1].append(vid)
    # ADD Self loop cause troubles due to heterogeneous type
    # ess = [add_self_loops(torch.tensor(es).long())[0] for es in ess]
    ess = [torch.tensor(es).long() for es in ess]

    data = Data(ess=ess)
    l_a = torch.tensor([
        meta_data.ntype2id[nx_g.nodes[n]['ntype']] for n in n2id['ast']
    ]).long()
    if ast_enc is not None:
        data.c_a = torch.tensor([
            ast_enc(nx_g.nodes[n]['token']) for n in n2id['ast']
        ]).float()
    # for cfg, it will be text
    data.lbl = torch.tensor([nx_g.nodes[n]['status'] for n in n2id['ast']])
    ts = torch.tensor([0] * (len(nx_g.nodes()) - len(n2id['ast'])))
    data.xs = [l_a, ts]
    return data, \
        torch.tensor(list(n2id['ast'][n] for n in stmt_nodes)).int()


def nx_to_pyg_node(meta_data, nx_g, ast_enc):
    nx_g = augment_with_reverse_edge_cat(nx_g, meta_data.t_e_asts, [])
    n2id = get_node2id(nx_g)
    ess = [[[], []] for i in range(len(meta_data.t_all))]
    for u, v, e in nx_g.edges(data=True):
        et_idx = meta_data.t_all.index(e['label'].replace('test', 't'))
        ess[et_idx][0].append(n2id[nx_g.nodes[u]['graph']][u])
        ess[et_idx][1].append(n2id[nx_g.nodes[v]['graph']][v])

    # ess = [add_self_loops(torch.tensor(es).long())[0] for es in ess]
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
    return data


class PyGStatementDataset(Dataset):

    def __init__(
            self, dataloader: NxDataloader,
            meta_data: AstGraphMetadata,
            ast_enc=None,
            save_dir='',
            name='pyg_cfl_stmt'
            ):
        self.dataloader = dataloader
        self.meta_data = meta_data if meta_data else\
            AstGraphMetadata(dataloader.get_dataset())
        self.save_dir = save_dir
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
        return nx_to_pyg_stmt(meta_data, nx_g, ast_enc, stmt_nodes)

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
