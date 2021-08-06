from __future__ import print_function, unicode_literals
import dgl
import torch
import torch.nn as nn
from layers.layers import WeightedGCN
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConstantNodeWeighters(torch.nn.Module):
    def __init__(self, val: float = 1.0):
        self.val = val

    def forward(self, h_g):
        # Make weighted features
        for ntype in h_g.ntypes:
            v = h_g.number_of_nodes(ntype)
            if v > 0:
                weight = Variable(
                    torch.tensor(
                        [self.val] * v).unsqueeze(-1).float(),
                    device=device, requires_grad=False)
                h_g.nodes[ntype].data['weight'] = weight
                h_g.nodes[ntype].data['h'] = h_g.nodes[ntype].data['h'] * \
                    self.weight_dict[ntype]
        return h_g


class ConstantEdgeWeighters(torch.nn.Module):
    def __init__(self, meta_graph):
        self.meta_graph = meta_graph

    def forward(self, h_g):
        for cetype in self.meta_graph:
            v = h_g.number_of_edges(cetype)
            if v > 0:
                weight = Variable(
                    torch.tensor(
                        [self.val] * v).unsqueeze(-1).float(),
                    device=device, requires_grad=False)
                h_g.edges[cetype].data['weight'] = weight
        return h_g


class ResGCN1TestNodeType(torch.nn.Module):
    def __init__(self, n_c_lbls: int, n_c_content_dim: int,
                 n_a_lbls: int, n_a_content_dim: int,
                 n_dim, meta_graph, n_o_classes, n_a_classes: int = 3,
                 device=device):
        super().__init__()

        self.c_lbl_emb = nn.Embedding(n_c_lbls, n_dim//2)
        self.c_c_enc = nn.Linear(n_c_content_dim, n_dim//2)
        nn.init.xavier_normal_(self.c_lbl_emb.weight)
        nn.init.xavier_normal_(self.c_c_enc.weight)

        self.a_lbl_emb = nn.Embedding(n_a_lbls, n_dim//2)
        self.a_c_enc = nn.Linear(n_a_content_dim, n_dim//2)

        nn.init.xavier_normal_(self.a_lbl_emb.weight)
        nn.init.xavier_normal_(self.a_c_enc.weight)
        nn.init.normal_(self.a_c_enc.bias)

        self.t_emb = nn.Parameter(torch.FloatTensor(n_dim))
        nn.init.normal_(self.test_embedding)

        self.meta_graph = meta_graph

        self.node_weighter = ConstantNodeWeighters(1.0)
        self.edge_weighter = ConstantEdgeWeighters(self.meta_graph)

        self.h_process1 = WeightedGCN(self.meta_graph, n_dim, n_dim, device)

        self.h_process2 = WeightedGCN(self.meta_graph, n_dim, n_dim, device)

        self.h_process3 = WeightedGCN(self.meta_graph, n_dim, n_dim, device)

        self.h_process4 = WeightedGCN(self.meta_graph, n_dim, n_dim, device)

        self.h_process5 = WeightedGCN(self.meta_graph, n_dim, n_dim, device)

        self.decoder = torch.nn.Linear(n_dim, n_o_classes)
        self.ast_decoder = torch.nn.Linear(n_dim, n_a_classes)

    def decode_node_func(self, nodes):
        feats = self.decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.last_act(feats)
        }

    def ast_decode_node_func(self, nodes):
        feats = self.ast_decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.last_act(feats)
        }

    def forward(self, h_g):
        h_g.nodes['cfg'].data['h'] = torch.cat((
            self.c_lbl_emb(h_g.nodes['cfg'].data['label']),
            self.c_c_enc(h_g.nodes['cfg'].data['content'].float())), dim=-1)

        h_g.nodes['ast'].data['h'] = torch.cat((
            self.a_lbl_emb(h_g.nodes['ast'].data['label']),
            self.a_c_enc(h_g.nodes['ast'].data['content'].float())),
            dim=-1)

        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = torch.cat(
                h_g.number_of_nodes('test') * [self.t_emb.unsqueeze(0)])

        h_g = self.node_weighter(h_g)
        h_g = self.edge_weighter(h_g)

        h_g = self.h_process1(h_g)
        cfg_feats = h_g.nodes['cfg'].data['h']
        ast_feats = h_g.nodes['ast'].data['h']

        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process2(h_g)
        h_g.nodes['cfg'].data['h'] += cfg_feats

        h_g.nodes['ast'].data['h'] += ast_feats

        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] += test_feats

        h_g = self.h_process3(h_g)

        cfg_feats = h_g.nodes['cfg'].data['h']
        ast_feats = h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process4(h_g)

        h_g.nodes['cfg'].data['h'] += cfg_feats
        h_g.nodes['ast'].data['h'] += ast_feats
        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] += test_feats

        h_g = self.h_process5(h_g)
        h_g.apply_nodes(self.decode_node_func, ntype='cfg')
        h_g.apply_nodes(self.ast_decode_node_func, ntype='ast')
        return h_g
