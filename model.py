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
                 num_classes, device=device,
                 ast_label_feats=None, ast_content_feats=None):
        super(HeteroMPNNPredictor, self).__init__()
        # Passing test overlapp
        # Failing test: often only one, so more important!
        # Intuition is failing test is rare, so it might contains more information
        # Similar to TD-IDF intuition

        self.cfg_label_encoder = nn.Linear(cfg_label_feats, hidden_feats//2)
        self.cfg_content_encoder = nn.Linear(cfg_content_feats, hidden_feats//2)

        if ast_label_feats != None and ast_content_feats != None:
            self.ast_label_encoder = nn.Linear(ast_label_feats, hidden_feats//2)
            self.ast_content_encoder = nn.Linear(ast_content_feats, hidden_feats//2)
            nn.init.xavier_normal_(self.ast_label_encoder.weight)
            nn.init.xavier_normal_(self.ast_label_encoder.bias)
            nn.init.xavier_normal_(self.ast_content_encoder.weight)
            nn.init.xavier_normal_(self.ast_content_encoder.bias)
        else:
            self.ast_label_encoder = None
            self.ast_content_encoder = None

        self.ptest_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal(self.ptest_embedding)

        self.ftest_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal(self.ftest_embedding)

        self.meta_graph = meta_graph
        '''
        self.etypes_params = torch.nn.ParameterDict()
        for cetype in self.meta_graph:
            etype = cetype[1]
            self.etypes_params[etype] = nn.Parameter(
                torch.FloatTensor(hidden_efeats))
        '''

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
            hidden_efeats, hidden_feats, device)

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
            'pred': self.last_act(feats)
        }


    def forward(self, h_g):
        # GNN creation on the fly
        h_g.nodes['cfg'].data['h'] = torch.cat((
            self.cfg_label_encoder(h_g.nodes['cfg'].data['label'].float()),
            self.cfg_content_encoder(h_g.nodes['cfg'].data['content'].float())),
            dim=-1)

        if self.ast_label_encoder != None and self.ast_content_encoder != None:
            h_g.nodes['ast'].data['h'] = torch.cat((
                self.ast_label_encoder(h_g.nodes['ast'].data['label'].float()),
                self.ast_content_encoder(h_g.nodes['ast'].data['content'].float())),
                dim=-1)

        if h_g.number_of_nodes('passing_test') > 0:
            h_g.nodes['passing_test'].data['h'] = torch.cat(
                h_g.number_of_nodes('passing_test') * [self.ptest_embedding.unsqueeze(0)])

        if h_g.number_of_nodes('failing_test') > 0:
            h_g.nodes['failing_test'].data['h'] = torch.cat(
                h_g.number_of_nodes('failing_test') * [self.ftest_embedding.unsqueeze(0)])

        '''
        for cetype in self.meta_graph:
            etype = cetype[1]
            h_g.edges[etype].data['h'] = torch.cat(
                    h_g.number_of_edges(etype) * [self.etypes_params[etype].unsqueeze(0)])
        '''

        # Let's cache stuffs here
        # Passing message
        h_g = self.h_process1(h_g)
        cfg_feats = h_g.nodes['cfg'].data['h']
        if self.ast_label_encoder != None and self.ast_content_encoder != None:
            ast_feats = h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('passing_test') > 0:
            ptest_feats = h_g.nodes['passing_test'].data['h']
        if h_g.number_of_nodes('failing_test') > 0:
            ftest_feats = h_g.nodes['failing_test'].data['h']

        h_g = self.h_process2(h_g)
        h_g.nodes['cfg'].data['h'] = torch.cat((
            cfg_feats, h_g.nodes['cfg'].data['h']), -1)

        if self.ast_label_encoder != None and self.ast_content_encoder != None:
            h_g.nodes['ast'].data['h'] = torch.cat((
                ast_feats, h_g.nodes['ast'].data['h']), -1)
        if h_g.number_of_nodes('passing_test') > 0:
            h_g.nodes['passing_test'].data['h'] = torch.cat((
                ptest_feats, h_g.nodes['passing_test'].data['h']), -1)
        if h_g.number_of_nodes('failing_test') > 0:
            h_g.nodes['failing_test'].data['h'] = torch.cat((
                ftest_feats, h_g.nodes['failing_test'].data['h']), -1)

        h_g = self.h_process3(h_g)

        cfg_feats = h_g.nodes['cfg'].data['h']
        if self.ast_label_encoder != None and self.ast_content_encoder != None:
            ast_feats = h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('passing_test') > 0:
            ptest_feats = h_g.nodes['passing_test'].data['h']
        if h_g.number_of_nodes('failing_test') > 0:
            ftest_feats = h_g.nodes['failing_test'].data['h']

        h_g = self.h_process4(h_g)

        h_g.nodes['cfg'].data['h'] = torch.cat((
            cfg_feats, h_g.nodes['cfg'].data['h']), -1)
        if self.ast_label_encoder != None and self.ast_content_encoder != None:
            h_g.nodes['ast'].data['h'] = torch.cat((
                ast_feats, h_g.nodes['ast'].data['h']), -1)
        if h_g.number_of_nodes('passing_test') > 0:
            h_g.nodes['passing_test'].data['h'] = torch.cat((
                ptest_feats, h_g.nodes['passing_test'].data['h']), -1)
        if h_g.number_of_nodes('failing_test') > 0:
            h_g.nodes['failing_test'].data['h'] = torch.cat((
                ftest_feats, h_g.nodes['failing_test'].data['h']), -1)

        h_g = self.h_process5(h_g)
        h_g.apply_nodes(self.decode_node_func, ntype='cfg')
        if self.ast_label_encoder != None and self.ast_content_encoder != None:
            h_g.apply_nodes(self.decode_node_func, ntype='ast')
        return h_g
