from __future__ import print_function, unicode_literals

import dgl
import torch
import torch.nn as nn
from layers.layers import HeteroMPNNBlockSimp


__author__ = "Marc: thanhdatn@student.unimelb.edu.au"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

edge_types = ['tb', 'lr', 'bt', 'child', 'parent', 'master']
e2idmap = [(e, i) for i, e in enumerate(edge_types)]


class HeteroMPNNPredictor(torch.nn.Module):
    def __init__(self, cfg_label_feats, cfg_content_feats,
                 hidden_feats, hidden_efeats, meta_graph,
                 num_classes,
                 device=device):
        super(HeteroMPNNPredictor, self).__init__()
        self.cfg_label_encoder = nn.Linear(cfg_label_feats, hidden_feats//2)
        self.cfg_content_encoder = nn.Linear(cfg_content_feats, hidden_feats//2)
        self.ptest_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        self.ftest_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))

        self.meta_graph = meta_graph
        self.etypes_params = torch.nn.ParameterDict()
        for cetype in self.meta_graph:
            etype = cetype[1]
            self.etypes_params[etype] = nn.Parameter(
                torch.FloatTensor(hidden_efeats))

        self.h_process1 = HeteroMPNNBlockSimp(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process2 = HeteroMPNNBlockSimp(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process3 = HeteroMPNNBlockSimp(
            self.meta_graph, hidden_feats*2,
            hidden_efeats, hidden_feats*2, device)
        self.h_process4 = HeteroMPNNBlockSimp(
            self.meta_graph, hidden_feats*2,
            hidden_efeats, hidden_feats*2, device)

        self.h_process5 = HeteroMPNNBlockSimp(
            self.meta_graph, hidden_feats*4,
            hidden_efeats, hidden_feats*4, device)

        self.decoder = torch.nn.Linear(hidden_feats, num_classes)
        if num_classes > 1:
            self.last_act = torch.nn.Softmax(dim=1)
        else:
            self.last_act = torch.nn.Sigmoid()

        self.device = device
        self.to(device)

    def decode_node_func(self, nodes):
        feats = self.decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.softmax(feats)
        }


    def forward(self, h_g):
        # GNN creation on the fly
        h_g.nodes['cfg'].data['h'] = torch.cat((
            self.cfg_label_encoder(h_g.nodes['cfg'].data['label'].float()),
            self.cfg_content_encoder(h_g.nodes['cfg'].data['content'].float())),
            dim=-1)
        h_g.nodes['passing_test'].data['h'] = torch.cat(
            h_g.number_of_nodes('passing_test') * [self.ptest_embedding.unsqueeze(0)])

        h_g.nodes['failing_test'].data['h'] = torch.cat(
            h_g.number_of_nodes('failing_test') * [self.ftest_embedding.unsqueeze(0)])

        for cetype in self.meta_graph:
            etype = cetype[1]
            h_g.edges[etype].data['h'] = torch.cat(
                    h_g.number_of_edges(etype) * [self.etypes_params[etype].unsqueeze(0)])

        # Let's cache stuffs here
        # Passing message
        h_g = self.h_process1(h_g)
        cfg_feats = h_g.nodes['cfg'].data['h']
        ptest_feats = h_g.nodes['passing_test'].data['h']
        ftest_feats = h_g.nodes['failing_test'].data['h']

        h_g = self.h_process2(h_g)
        h_g.nodes['cfg'].data['h'] = torch.cat((
            cfg_feats, h_g.nodes['cfg'].data['h']), -1)
        h_g.nodes['passing_test'].data['h'] = torch.cat((
            ptest_feats, h_g.nodes['passing_test'].data['h']), -1)
        h_g.nodes['failing_test'].data['h'] = torch.cat((
            ftest_feats, h_g.nodes['failing_test'].data['h']), -1)

        h_g = self.h_process3(h_g)

        cfg_feats = h_g.nodes['cfg'].data['h']
        ptest_feats = h_g.nodes['passing_test'].data['h']
        ftest_feats = h_g.nodes['failing_test'].data['h']

        h_g = self.h_process4(h_g)

        h_g.nodes['cfg'].data['h'] = torch.cat((
            cfg_feats, h_g.nodes['cfg'].data['h']), -1)
        h_g.nodes['passing_test'].data['h'] = torch.cat((
            ptest_feats, h_g.nodes['passing_test'].data['h']), -1)
        h_g.nodes['failing_test'].data['h'] = torch.cat((
            ftest_feats, h_g.nodes['failing_test'].data['h']), -1)

        h_g = self.h_process5(h_g)
        h_g.apply_nodes(self.decode_node_func, ntype='cfg')
        return h_g
