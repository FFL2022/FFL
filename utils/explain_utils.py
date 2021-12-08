import networkx as nx
from utils.nx_graph_builder import augment_with_reverse_edge

import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import math
import copy
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def map_explain_with_nx(dgl_g, nx_g):
    # check every types of edges
    n_as = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
    n_cs = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'cfg']
    n_ts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']

    n_alls = {'ast': n_as, 'cfg': n_cs, 'test': n_ts}

    # print(len(n_alls['ast']))
    # print(dgl_g.nodes(ntype='ast').shape)
    # exit()

    # Augment with reverse edge so that the two met
    # ast_etypes = set()
    # for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
    #     if nx_g.nodes[u]['graph'] == 'ast' and\
    #             nx_g.nodes[v]['graph'] == 'ast':
    #         ast_etypes.add(e['label'])
    # cfg_etypes = set()
    # for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
    #     if nx_g.nodes[u]['graph'] == 'cfg' and\
    #             nx_g.nodes[v]['graph'] == 'cfg':
    #         cfg_etypes.add(e['label'])

    # nx_g = augment_with_reverse_edge(nx_g, ast_etypes, cfg_etypes)

    all_etypes = set()
    for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
        all_etypes.add((nx_g.nodes[u]['graph'],
                        e['label'],
                        nx_g.nodes[v]['graph']))

    existed_etypes = []
    for etype in dgl_g.etypes:
        if dgl_g.number_of_edges(etype) > 0:
            existed_etypes.append(etype)
    existed_etypes = list(set(existed_etypes))
    # print(all_etypes)
    # print(existed_etypes)


    # Loop through each type of edges
    for etype in all_etypes:
        if etype[1] not in existed_etypes:
            continue
        # Get edge in from dgl
        # print(etype, type(etype))
        # exit()
        # if dgl_g.number_of_edges(etype) == 0:
        #     continue
        es = dgl_g.edges(etype=etype)
        data = dgl_g.edges[etype].data['weight']
        # print(data.min(), data.max(), data.mean())

        # magic
        data = data * 7

        # print(es)
        # if 'weight' not in es.data:
        #     continue
        us = es[0]
        vs = es[1]
        for i in range(us.shape[0]):
            u = n_alls[etype[0]][us[i].item()]
            v = n_alls[etype[2]][vs[i].item()]
            # print(f'u={u}, v={v}, vs[{i}]={vs[i]}')
            for k in nx_g[u][v]:
                nx_g[u][v][k]['penwidth'] = data[i].item()
    # Loop through each type of node
    all_ntypes = set([et[0] for et in all_etypes]).union(
        set([et[1] for et in all_etypes]))
    existed_ntypes = set([ntype for ntype in dgl_g.ntypes
                      if dgl_g.number_of_nodes(ntype) > 0]).intersection(all_ntypes)
    for ntype in existed_ntypes:
        ns = dgl_g.nodes(ntype)
        n_w = dgl_g.nodes[ntype].data['weight']
        # print(ns.shape, n_w.shape)
        # exit()
        # magic
        n_w = n_w * 7
        for i in range(ns.shape[0]):
            nx_g.nodes[n_alls[ntype][i]]['penwidth'] = n_w[i].item()

    return  nx_g



class EdgeWeights(nn.Module):
    def __init__(self, num_edges, etype):
        super(EdgeWeights, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.params = nn.Parameter(torch.FloatTensor(num_edges).unsqueeze(-1).to(device))
        nn.init.normal_(self.params, nn.init.calculate_gain("relu")*math.sqrt(2.0)/(num_edges*2))

        self.etype = etype

        # self.temp = self.params.clone()

    def forward(self, g):
        # if not torch.all(self.temp == self.params):
        #     print('[+] Edge', self.etype, torch.all(self.temp == self.params))
        # self.temp = self.params.clone()
        g.edges[self.etype].data['weight'] = self.sigmoid(self.params)
        return g

class NodeWeights(nn.Module):

    def __init__(self, num_nodes, num_node_feats):
        super(NodeWeights, self).__init__()
        self.params = nn.Parameter(
            torch.FloatTensor(num_nodes, 1).to(device))
        nn.init.normal_(self.params, nn.init.calculate_gain(
            "relu")*math.sqrt(2.0)/(num_nodes*2))
        self.sigmoid = nn.Sigmoid()
        # self.temp = self.params.clone()

    def forward(self, g):
        # if not torch.all(self.temp == self.params):
        #     print('[+] Node', torch.all(self.temp == self.params))
        # self.temp = self.params.clone()
        # print(self.params[0][0])
        g.nodes['ast'].data['weight'] = self.sigmoid(self.params)
        return g

class HeteroGraphWeights(nn.Module):

    def __init__(self, num_nodes, num_edges_dict, num_node_feats):
        super(HeteroGraphWeights, self).__init__()
        self.nweights = NodeWeights(num_nodes, num_node_feats)
        self.eweights = nn.ModuleDict()
        for etype in num_edges_dict:
            if etype == 'type':
                self.eweights['_type'] = EdgeWeights(num_edges_dict[etype], etype)
            else:
                self.eweights[etype] = EdgeWeights(num_edges_dict[etype], etype)

        # self.eweights = EdgeWeights(num_edges_dict['parent_child'], 'parent_child')
        self.etypes = list(num_edges_dict.keys())

    def forward(self, g):
        g = self.nweights(g)
        for etype in self.etypes:
            if etype == 'type':
                etype = '_type'
            g = self.eweights[etype](g)
        # g = self.eweights(g)
        return g


class WrapperModel(nn.Module):
    def __init__(self, model, num_nodes, num_edges_dict, num_node_feats):
        super(WrapperModel, self).__init__()
        self.hgraph_weights = HeteroGraphWeights(num_nodes, num_edges_dict, num_node_feats)
        self.model = model

    def forward_old(self, g):
        self.model.eval()
        self.model.add_default_nweight = True
        self.model.add_default_eweight = True
        return self.model(g).nodes['ast'].data['logits']

    def forward(self, g):
        self.model.eval()
        self.model.add_default_nweight = False
        self.model.add_default_eweight = False

        g = self.hgraph_weights(g)
        return self.model(g).nodes['ast'].data['logits']


def entropy_loss(masking):
    return torch.mean(
        -torch.sigmoid(masking) * torch.log(torch.sigmoid(masking)) -
        (1 - torch.sigmoid(masking)) * torch.log(1 - torch.sigmoid(masking)))


def entropy_loss_mask(g, etypes, coeff_n=0.1, coeff_e=0.3):
    e_e_loss = 0
    for etype in etypes:
        e_e_loss += entropy_loss(g.edges[etype].data['weight'])
    e_e_loss = coeff_e * e_e_loss
    n_e_loss = coeff_n * entropy_loss(g.nodes['ast'].data['weight'])
    return n_e_loss + e_e_loss

def consistency_loss(preds, labels):
    loss = F.cross_entropy(preds, labels)
    return loss



def size_loss(g, etypes, coeff_n=0.002, coeff_e=0.005):
    feat_size_loss = coeff_n * torch.sum(g.nodes['ast'].data['weight'])
    edge_size_loss = 0
    for etype in etypes:
        edge_size_loss += g.edges[etype].data['weight'].sum()
    edge_size_loss = coeff_e * edge_size_loss
    return feat_size_loss + edge_size_loss
