from __future__ import print_function, unicode_literals

import dgl
import torch
import torch.nn as nn
from dgl_version.layers.layers import GCNLayer, GCNLayerOld


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
        self.cfg_content_encoder = nn.Linear(
            cfg_content_feats, hidden_feats//2)

        if ast_label_feats is not None and ast_content_feats is not None:
            self.ast_label_encoder = nn.Linear(
                ast_label_feats, hidden_feats//2)
            self.ast_content_encoder = nn.Linear(
                ast_content_feats, hidden_feats//2)
            nn.init.xavier_normal_(self.ast_label_encoder.weight)
            nn.init.normal_(self.ast_label_encoder.bias)
            nn.init.xavier_normal_(self.ast_content_encoder.weight)
            nn.init.normal_(self.ast_content_encoder.bias)
        else:
            self.ast_label_encoder = None
            self.ast_content_encoder = None

        self.ptest_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal_(self.ptest_embedding)

        self.ftest_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal_(self.ftest_embedding)

        self.meta_graph = meta_graph
        '''
        self.etypes_params = torch.nn.ParameterDict()
        for cetype in self.meta_graph:
            etype = cetype[1]
            self.etypes_params[etype] = nn.Parameter(
                torch.FloatTensor(hidden_efeats))
        '''

        self.h_process1 = GCNLayerOld(self.meta_graph, hidden_feats,
                                      hidden_feats, device)

        self.h_process2 = GCNLayerOld(self.meta_graph, hidden_feats,
                                      hidden_feats, device)

        self.h_process3 = GCNLayerOld(self.meta_graph, hidden_feats*2,
                                      hidden_feats*2, device)
        self.h_process4 = GCNLayerOld(self.meta_graph, hidden_feats*2,
                                      hidden_feats*2, device)

        self.h_process5 = GCNLayerOld(self.meta_graph, hidden_feats*4,
                                      hidden_feats, device)

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


class HeteroMPNNPredictor1TestNodeType(torch.nn.Module):
    def __init__(self, num_cfg_label, cfg_content_feats,
                 hidden_feats, hidden_efeats, meta_graph,
                 num_classes, device=device,
                 num_ast_labels=None, ast_content_feats=None, num_classes_ast=3):
        super().__init__()
        # Passing test overlapp
        # Failing test: often only one, so more important!
        # Intuition is failing test is rare, so it might contains more information
        # Similar to TD-IDF intuition

        self.cfg_label_encoder = nn.Embedding(num_cfg_label, hidden_feats//2)
        self.cfg_content_encoder = nn.Linear(
            cfg_content_feats, hidden_feats//2)

        if num_ast_labels is not None and ast_content_feats is not None:
            self.ast_label_encoder = nn.Embedding(
                num_ast_labels, hidden_feats//2)
            self.ast_content_encoder = nn.Linear(
                ast_content_feats, hidden_feats//2)
            nn.init.xavier_normal_(self.ast_label_encoder.weight)
            nn.init.xavier_normal_(self.ast_content_encoder.weight)
            nn.init.normal_(self.ast_content_encoder.bias)
        else:
            self.ast_label_encoder = None
            self.ast_content_encoder = None

        self.test_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal_(self.test_embedding)

        self.meta_graph = meta_graph

        self.h_process1 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process2 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process3 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)
        self.h_process4 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process5 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.decoder = torch.nn.Linear(hidden_feats, num_classes)
        self.ast_decoder = torch.nn.Linear(hidden_feats, num_classes_ast)
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

    def ast_decode_node_func(self, nodes):
        feats = self.ast_decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.last_act(feats)
        }

    def forward(self, h_g):
        '''
        # GNN creation on the fly
        if (h_g.number_of_nodes('cfg') > 0):
            h_g.nodes['cfg'].data['h'] = torch.cat((
                self.cfg_label_encoder(h_g.nodes['cfg'].data['label']),
                self.cfg_content_encoder(
                    h_g.nodes['cfg'].data['content'].float())),
                dim=-1)
                '''

        if self.ast_label_encoder is not None and\
                self.ast_content_encoder != None:
            h_g.nodes['ast'].data['h'] = torch.cat((
                self.ast_label_encoder(h_g.nodes['ast'].data['label']),
                self.ast_content_encoder(h_g.nodes['ast'].data['content'].float())),
                dim=-1)

        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = torch.cat(
                h_g.number_of_nodes('test') * [self.test_embedding.unsqueeze(0)])

        # Let's cache stuffs here
        # Passing message
        h_g = self.h_process1(h_g)
        '''
        if (h_g.number_of_nodes('cfg') > 0):
            cfg_feats = h_g.nodes['cfg'].data['h']
            '''
        if self.ast_label_encoder is not None and self.ast_content_encoder != None:
            ast_feats = h_g.nodes['ast'].data['h']

        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process2(h_g)
        '''
        if (h_g.number_of_nodes('cfg') > 0):
            h_g.nodes['cfg'].data['h'] = torch.cat((
                cfg_feats, h_g.nodes['cfg'].data['h']), -1)
                '''

        if self.ast_label_encoder is not None and self.ast_content_encoder != None:
            h_g.nodes['ast'].data['h'] = ast_feats + h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = test_feats + \
                h_g.nodes['test'].data['h']

        h_g = self.h_process3(h_g)

        '''
        if (h_g.number_of_nodes('cfg') > 0):
            cfg_feats = h_g.nodes['cfg'].data['h']
            '''
        if self.ast_label_encoder is not None and self.ast_content_encoder != None:
            ast_feats = h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process4(h_g)

        '''
        if (h_g.number_of_nodes('cfg') > 0):
            h_g.nodes['cfg'].data['h'] = torch.cat((
                cfg_feats, h_g.nodes['cfg'].data['h']), -1)
                '''
        if self.ast_label_encoder is not None and self.ast_content_encoder != None:
            h_g.nodes['ast'].data['h'] = ast_feats + h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = test_feats + \
                h_g.nodes['test'].data['h']

        h_g = self.h_process5(h_g)
        '''
        if (h_g.number_of_nodes('cfg') > 0):
            h_g.apply_nodes(self.decode_node_func, ntype='cfg')
            '''
        if self.ast_label_encoder is not None and self.ast_content_encoder != None:
            h_g.apply_nodes(self.ast_decode_node_func, ntype='ast')
        return h_g


class HeteroMPNNPredictor1TestNodeTypeArity(torch.nn.Module):
    def __init__(self, num_cfg_label, cfg_content_feats,
                 hidden_feats, hidden_efeats, meta_graph,
                 max_ast_arity,
                 num_classes, device=device,
                 num_ast_labels=None, ast_content_feats=None, num_classes_ast=3):
        super().__init__()
        # Passing test overlapp
        # Failing test: often only one, so more important!
        # Intuition is failing test is rare, so it might contains more information
        # Similar to TD-IDF intuition

        self.cfg_label_encoder = nn.Embedding(num_cfg_label, hidden_feats//2)
        self.cfg_content_encoder = nn.Linear(
            cfg_content_feats, hidden_feats//2)

        self.ast_arity_encoder = nn.Embedding(max_ast_arity, hidden_feats//2)
        self.ast_label_encoder = nn.Embedding(
            num_ast_labels, hidden_feats//2)
        self.ast_content_encoder = nn.Linear(
            ast_content_feats, hidden_feats//2)
        nn.init.xavier_normal_(self.ast_label_encoder.weight)
        nn.init.xavier_normal_(self.ast_content_encoder.weight)
        nn.init.normal_(self.ast_content_encoder.bias)

        self.test_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal_(self.test_embedding)

        self.meta_graph = meta_graph

        self.h_process1 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process2 = GCNLayer(
            self.meta_graph, hidden_feats,
            hidden_efeats, hidden_feats, device)

        self.h_process3 = GCNLayer(
            self.meta_graph, hidden_feats*2,
            hidden_efeats, hidden_feats*2, device)
        self.h_process4 = GCNLayer(
            self.meta_graph, hidden_feats*2,
            hidden_efeats, hidden_feats*2, device)

        self.h_process5 = GCNLayer(
            self.meta_graph, hidden_feats*4,
            hidden_efeats, hidden_feats, device)

        self.decoder = torch.nn.Linear(hidden_feats, num_classes)
        self.ast_decoder = torch.nn.Linear(hidden_feats, num_classes_ast)
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

    def ast_decode_node_func(self, nodes):
        feats = self.ast_decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.last_act(feats)
        }

    def forward(self, h_g):
        # GNN creation on the fly
        h_g.nodes['cfg'].data['h'] = torch.cat((
            self.cfg_label_encoder(h_g.nodes['cfg'].data['label']),
            self.cfg_content_encoder(
                h_g.nodes['cfg'].data['content'].float())),
            dim=-1)

        h_g.nodes['ast'].data['h'] = torch.cat((
            self.ast_label_encoder(h_g.nodes['ast'].data['label']) +
            self.ast_arity_encoder(h_g.nodes['ast'].data['arity']),
            self.ast_content_encoder(h_g.nodes['ast'].data['content'].float())),
            dim=-1)

        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = torch.cat(
                h_g.number_of_nodes('test') * [self.test_embedding.unsqueeze(0)])

        # Let's cache stuffs here
        # Passing message
        h_g = self.h_process1(h_g)
        cfg_feats = h_g.nodes['cfg'].data['h']
        ast_feats = h_g.nodes['ast'].data['h']

        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process2(h_g)
        h_g.nodes['cfg'].data['h'] = torch.cat((
            cfg_feats, h_g.nodes['cfg'].data['h']), -1)

        h_g.nodes['ast'].data['h'] = torch.cat((
            ast_feats, h_g.nodes['ast'].data['h']), -1)
        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = torch.cat((
                test_feats, h_g.nodes['test'].data['h']), -1)

        h_g = self.h_process3(h_g)

        cfg_feats = h_g.nodes['cfg'].data['h']
        ast_feats = h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process4(h_g)

        h_g.nodes['cfg'].data['h'] = torch.cat((
            cfg_feats, h_g.nodes['cfg'].data['h']), -1)
        h_g.nodes['ast'].data['h'] = torch.cat((
            ast_feats, h_g.nodes['ast'].data['h']), -1)
        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = torch.cat((
                test_feats, h_g.nodes['test'].data['h']), -1)

        h_g = self.h_process5(h_g)
        h_g.apply_nodes(self.decode_node_func, ntype='cfg')
        h_g.apply_nodes(self.ast_decode_node_func, ntype='ast')
        return h_g


class GCN_A_L_T_1(torch.nn.Module):
    def __init__(self, hidden_feats, meta_graph,
                 device=device, num_ast_labels=None, num_classes_ast=3):
        super().__init__()
        # Passing test overlapp
        # Failing test: often only one, so more important!
        # Intuition is failing test is rare, so it might contains more information
        # Similar to TD-IDF intuition
        self.ast_label_encoder = nn.Embedding(
            num_ast_labels, hidden_feats)
        nn.init.xavier_normal_(self.ast_label_encoder.weight)

        self.test_embedding = nn.Parameter(torch.FloatTensor(hidden_feats))
        nn.init.normal_(self.test_embedding)

        self.meta_graph = meta_graph

        self.h_process1 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process2 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process3 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process4 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process5 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.ast_decoder = torch.nn.Linear(hidden_feats, num_classes_ast)
        self.last_act = torch.nn.Softmax(dim=1)

        self.device = device
        self.to(device)

    def ast_decode_node_func(self, nodes):
        feats = self.ast_decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.last_act(feats)
        }

    def forward(self, h_g):
        h_g.nodes['ast'].data['h'] = self.ast_label_encoder(
            h_g.nodes['ast'].data['label'])
        h_g = self.node_weight_multiply(h_g)

        if self.add_default_eweight:
            for etype in h_g.etypes:
                h_g.edges[etype].data['weight'] = torch.Tensor([1] * \
                    h_g.number_of_edges(etype)).unsqueeze(-1).to(h_g.device)

        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = torch.cat(
                h_g.number_of_nodes('test') *
                [self.test_embedding.unsqueeze(0)])

        # Let's cache stuffs here
        # Passing message
        h_g = self.h_process1(h_g)
        if self.ast_label_encoder is not None:
            ast_feats = h_g.nodes['ast'].data['h']

        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process2(h_g)
        if self.ast_label_encoder is not None:
            h_g.nodes['ast'].data['h'] = ast_feats + h_g.nodes['ast'].data['h']

        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = test_feats + \
                h_g.nodes['test'].data['h']

        h_g = self.h_process3(h_g)

        if self.ast_label_encoder is not None:
            ast_feats = h_g.nodes['ast'].data['h']

        if h_g.number_of_nodes('test') > 0:
            test_feats = h_g.nodes['test'].data['h']

        h_g = self.h_process4(h_g)

        if self.ast_label_encoder is not None:
            h_g.nodes['ast'].data['h'] = ast_feats + h_g.nodes['ast'].data['h']
        if h_g.number_of_nodes('test') > 0:
            h_g.nodes['test'].data['h'] = test_feats + \
                h_g.nodes['test'].data['h']

        h_g = self.h_process5(h_g)
        if self.ast_label_encoder is not None:
            h_g.apply_nodes(self.ast_decode_node_func, ntype='ast')
        return h_g


class GCN_A_L(torch.nn.Module):
    def __init__(self, hidden_feats, meta_graph,
                 device=device, num_ast_labels=None, num_classes_ast=3):
        super().__init__()
        self.ast_label_encoder = nn.Embedding(
            num_ast_labels, hidden_feats)
        nn.init.xavier_normal_(self.ast_label_encoder.weight)

        self.meta_graph = meta_graph

        self.h_process1 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process2 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process3 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process4 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.h_process5 = GCNLayer(
            self.meta_graph, hidden_feats, hidden_feats, device)

        self.ast_decoder = torch.nn.Linear(hidden_feats, num_classes_ast)
        self.last_act = torch.nn.Softmax(dim=1)

        self.device = device
        self.to(device)

    def ast_decode_node_func(self, nodes):
        feats = self.ast_decoder(nodes.data['h'])
        return {
            'logits': feats,
            'pred': self.last_act(feats)
        }

    def forward(self, h_g):
        h_g.nodes['ast'].data['h'] = self.ast_label_encoder(
            h_g.nodes['ast'].data['label'])
        #print(h_g.nodes['ast'].data['h'])
        # Let's cache stuffs here
        # Passing message
        h_g = self.h_process1(h_g)
        if self.ast_label_encoder is not None:
            ast_feats = h_g.nodes['ast'].data['h']

        h_g = self.h_process2(h_g)
        if self.ast_label_encoder is not None:
            h_g.nodes['ast'].data['h'] = ast_feats + h_g.nodes['ast'].data['h']

        h_g = self.h_process3(h_g)

        if self.ast_label_encoder is not None:
            ast_feats = h_g.nodes['ast'].data['h']

        h_g = self.h_process4(h_g)

        if self.ast_label_encoder is not None:
            h_g.nodes['ast'].data['h'] = ast_feats + h_g.nodes['ast'].data['h']

        h_g = self.h_process5(h_g)
        return h_g
